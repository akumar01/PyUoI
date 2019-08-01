import numpy as np
import pdb
from .base import AbstractUoILinearRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model.coordinate_descent import _alpha_grid
from sklearn.linear_model import ElasticNet, RidgeCV
import pycasso

class UoI_ElasticNet(AbstractUoILinearRegressor, LinearRegression):
    """ UoI ElasticNet model.

    Parameters
    ----------
    n_boots_sel : int, default 48
        The number of data bootstraps to use in the selection module.
        Increasing this number will make selection more strict.

    n_boots_est : int, default 48
        The number of data bootstraps to use in the estimation module.
        Increasing this number will relax selection and decrease variance.

    selection_frac : float, default 0.9
        The fraction of the dataset to use for training in each resampled
        bootstrap, during the selection module. Small values of this parameter
        imply larger "perturbations" to the dataset.

    estimation_frac : float, default 0.9
        The fraction of the dataset to use for training in each resampled
        bootstrap, during the estimation module. The remaining data is used
        to obtain validation scores. Small values of this parameters imply
        larger "perturbations" to the dataset. IGNORED - Leaving this here
        to double check later

    n_lambdas : int, default 48
        The number of regularization values to use for selection.

    alphas : list or ndarray of floats
        The parameter that trades off L1 versus L2 regularization for a given
        lambda.

    stability_selection : int, float, or array-like, default 1
        If int, treated as the number of bootstraps that a feature must
        appear in to guarantee placement in selection profile. If float,
        must be between 0 and 1, and is instead the proportion of
        bootstraps. If array-like, must consist of either ints or floats
        between 0 and 1. In this case, each entry in the array-like object
        will act as a separate threshold for placement in the selection
        profile.

    estimation_score : str "r2" | "AIC", | "AICc" | "BIC"
        Objective used to choose the best estimates per bootstrap.

    warm_start : bool, default True
        When set to ``True``, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution

    eps : float, default 1e-3
        Length of the lasso path. eps=1e-3 means that
        alpha_min / alpha_max = 1e-3

    copy_X : boolean, default True
        If ``True``, X will be copied; else, it may be overwritten.

    fit_intercept : boolean, default True
        Whether to calculate the intercept for this model. If set
        to False, no intercept will be used in calculations
        (e.g. data is expected to be already centered).

    standardize : boolean, default False
        If True, the regressors X will be standardized before regression by
        subtracting the mean and dividing by their standard deviations.

    max_iter : int, default None
        Maximum number of iterations for iterative fitting methods.

    random_state : int, RandomState instance or None, default None
        The seed of the pseudo random number generator that selects a random
        feature to update.  If int, random_state is the seed used by the random
        number generator; If RandomState instance, random_state is the random
        number generator; If None, the random number generator is the
        RandomState instance used by `np.random`.

    comm : MPI communicator, default None
        If passed, the selection and estimation steps are parallelized.

    Attributes
    ----------
    coef_ : array, shape (n_features,) or (n_targets, n_features)
        Estimated coefficients for the linear regression problem.

    intercept_ : float
        Independent term in the linear model.

    supports_ : array, shape
        boolean array indicating whether a given regressor (column) is selected
        for estimation for a given regularization parameter value (row).
    """
    def __init__(self, n_boots_sel=48, n_boots_est=48, selection_frac=0.9,
                 estimation_frac=0.9, n_lambda1=48,
                 n_lambda2=5, stability_selection=1.,
                 estimation_score='r2', warm_start=True, eps=1e-3,
                 copy_X=True, fit_intercept=True, standardize=True,
                 max_iter=1000, random_state=None, comm=None, logger=None):
        super(UoI_ElasticNet, self).__init__(
            n_boots_sel=n_boots_sel,
            n_boots_est=n_boots_est,
            selection_frac=selection_frac,
            estimation_frac=estimation_frac,
            stability_selection=stability_selection,
            estimation_score=estimation_score,
            copy_X=copy_X,
            fit_intercept=fit_intercept,
            standardize=standardize,
            random_state=random_state,
            comm=comm,
            max_iter=max_iter,
            logger=logger
        )
        self.n_lambda1 = n_lambda1
        self.n_lambda2 = n_lambda2
        self.warm_start = warm_start
        self.eps = eps
        self._selection_lm = PycassoElasticNet(
            fit_intercept=fit_intercept,
            max_iter=max_iter)
        self._estimation_lm = LinearRegression(fit_intercept=fit_intercept)

    def get_reg_params(self, X, y):
        r"""Calculates the regularization parameters (lambda1 and lambda2) to be
        used for the provided data.

        Note that the Elastic Net penalty is given by

        .. math::
           \frac{1}{2\ \text{n_samples}} ||y - Xb||^2_2
           + lambda_1 |b|_1 + lambda_2 |b|^2_2)

        where lambda_1 and lambda_2 are regularization parameters.

        This parameterization allows for pathwise solutions. This differs
        from the paramterization that sklearn decides to use for its ElasticNet
        in terms of alpha and l1_ratio

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The design matrix.

        y : array-like, shape (n_samples)
            The response vector.

        Returns
        -------
        reg_params : a list of dictionaries
            A list containing dictionaries with the value of each
            (lambda1, lambda2) describing the type of regularization to impose.
        """

        # To sensibly limit the lambda2 grid, use sklearn's ridge regression to
        # calculate an upper bound on the penalty to be applied, and then logspace
        # from 0 to this upper penalty 

        # Set the search space for the ridge penalty by the largest l2 penalty
        # induced by an _alpha_grid search induced by a 0.1 l1_ratio
        l1_ = _alpha_grid(X, y, l1_ratio = 0.1)
        l2_max = np.max((1 - 0.1) * l1_/2)
        # Set the min by the smallest l2 penalty induced by an _alpha_grid search 
        # induced by a 0.99 l1_ratio
        l1_ = _alpha_grid(X, y, l1_ratio = 0.99)
        l2_min = np.min((1 - 0.99) * l1_/2)

        r = RidgeCV(np.linspace(l2_min, l2_max, 100), fit_intercept = self.fit_intercept)
        r.fit(X, y)
        l2_max = r.alpha_

        self.lambda2 = np.logspace(np.log10(l2_min), np.log10(l2_max), self.n_lambda2)

        # place the regularization parameters into a list of dictionaries, with a 
        # unique dictionary for each (l1, l2) pair. This is for compatibility with the 
        # base UoI module. Also record an array of l1 parameters for easy use in the 
        # path-wise pycasso solver

        reg_params = []
        lambda1 = np.zeros((self.n_lambda2, self.n_lambda1))

        for i, l2 in enumerate(self.lambda2):

            # For each lambda2, augment the data into an L1 problem and call _alpha_grid
            # to get the lambda1 values
            xx, yy = augment_data(X, y, l2)

            gamma = _alpha_grid(xx, yy, n_alphas = self.n_lambda1)

            # Transform these gamma values into the l1 penalties used in the Elastic Net
            # problem. This is done for transparency and consistency of notation
            lambda1_ = np.sqrt(1 + l2) * gamma
            lambda1[i, :] = lambda1_

            for l1 in lambda1_:
                reg_params.append([{'lambda1' : l1, 'lambda2' : l2}])

        self.lambda1 = lambda1

        return reg_params

    # Overwrite base class selection sweep to accomodate Pycasso 
    # path-wise solution
    def uoi_selection_sweep(self, X, y, reg_param_values):

        n_coef = self.get_n_coef(X, y)        
        coefs = np.zeros((self.n_lambda2 * self.n_lambda1, n_coef))

        for i, l2  in enumerate(self.lambda2):

            l1 = self.lambda1[i, :]

            self._selection_lm.init_solver(X, y, l1, l2)

            self._selection_lm.fit()
            coefs[i * self.n_lambda1 :(i + 1) * self.n_lambda1, :] = self._selection_lm.coef_

        return coefs 

# Pycasso solver wrapper with minimal class structure to interface with UoI
class PycassoElasticNet():

    def __init__(self, fit_intercept = False, max_iter = 1000):
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept

    def init_solver(self, X, y, lambda1 = None, lambda2 = None):

        # Set lambda2 using a RidgeCV fit
        if lambda2 is None:
            rdge = RidgeCV(alphas = np.linspace(1e-5, 20, 100)).fit(X, y)
            lambda2 = rdge.alpha_
        self.lambda2 = lambda2

        self.dummy_path = False

        if lambda1 is None:
            lambda1 = _alpha_grid(X, y, n_alphas = 100)
        else:
            if np.isscalar(lambda1):
                lambda1 = np.array([lambda1])
            lambda1 = np.flipud(np.sort(lambda1))
            if lambda1.size < 3:
                lambda1 = np.sort(lambda1)
                # Create a dummy path for the path solver
                self.dummy_path = True
                self.pathlength = lambda1.size
                while lambda1.size < 3:
                    lambda1 = np.append(lambda1, lambda1[-1]/2)
        self.lambda1 = lambda1

        # We solve for an entire elastic net path with a fixed lambda2
        # For the given fixed lambda1, we modify the dataset to allow 
        # for the solution of a lasso-like problem
        xx, yy = augment_data(X, y, self.lambda2)

        # Augmented regularization parameters
        gamma = self.lambda1/np.sqrt(1 + self.lambda2)
        self.solver = pycasso.Solver(xx, yy, family = 'gaussian', 
                      useintercept = self.fit_intercept, lambdas = gamma,
                      penalty = 'l1', max_ite = self.max_iter)

    def fit(self, X, y, lambda1 = None, lambda2 = None):
        self.init_solver(X, y, lambda1, lambda2)
        self.solver.train()
        # Coefs across the entire solution path
        beta_naive = self.solver.result['beta']

    
        if self.dummy_path:
            beta_naive = beta_naive[:self.pathlength, :]

        # Rescale coefficients (eq. 11 of Elastic Net paper)
        self.coef_ = np.sqrt(1 + self.lambda2) * beta_naive

        # Record regularization parameters
        reg_params = np.zeros((self.lambda1.size, 2))
        reg_params[:, 0] = self.lambda2
        reg_params[:, 1] = self.lambda1

        self.reg_params = reg_params

# Augment data so ElasticNet becomes an l1 regularization problem 
def augment_data(X, y, l2):

    n_samples, n_features = X.shape

    if y.ndim == 1:
        y = y[:, np.newaxis]

    # Augment the data
    XX = 1/np.sqrt(1 + l2) * np.vstack([X, np.sqrt(2 * n_samples) * np.sqrt(l2) * np.eye(n_features)])
    yy = np.vstack([y, np.zeros((n_features, 1))])
 
    return XX, yy