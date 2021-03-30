import numpy as np

from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.linear_model._coordinate_descent import _alpha_grid
try:
    import pycasso
except ImportError:
    pycasso = None
from .base import AbstractUoILinearRegressor, OLS_Wrapper
from .pyc import PycWrapper

class UoI_NCV(AbstractUoILinearRegressor, LinearRegression):
    r"""UoI\ :sub:`SCAD` or 'MCP' solver.

    Parameters
     ----------
    n_boots_sel : int
        The number of data bootstraps/resamples to use in the selection module.
        Increasing this number will make selection more strict.
    n_boots_est : int
        The number of data bootstraps/resamples to use in the estimation
        module. Increasing this number will relax selection and decrease
        variance.
    n_lambdas : int
        The number of regularization values to use for selection.
    selection_frac : float
        The fraction of the dataset to use for training in each resampled
        bootstrap, during the selection module. Small values of this parameter
        imply larger "perturbations" to the dataset.
    estimation_frac : float
        The fraction of the dataset to use for training in each resampled
        bootstrap, during the estimation module. The remaining data is used
        to obtain validation scores. Small values of this parameters imply
        larger "perturbations" to the dataset.
    stability_selection : int, float, or array-like
        If int, treated as the number of bootstraps that a feature must
        appear in order to guarantee placement in selection profile. If float,
        must be between 0 and 1, and is instead the proportion of
        bootstraps. If array-like, must consist of either ints or floats
        between 0 and 1. In this case, each entry in the array-like object
        will act as a separate threshold for placement in the selection
        profile.
    estimation_score : string, "r2" | "AIC" | "AICc" | "BIC"
        Objective used to choose the best estimates per bootstrap.
    estimation_target : string, "train" | "test"
        Decide whether to assess the estimation_score on the train
        or test data across each bootstrap. By deafult, a sensible
        choice is made based on the chosen estimation_score
    warm_start : bool
        When set to ``True``, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution
    eps : float
        Length of the lasso path. ``eps=1e-3`` means that
        ``lambda_min / lambda_max = 1e-3``
    copy_X : bool
        If ``True``, X will be copied; else, it may be overwritten.
    fit_intercept : bool
        Whether to calculate the intercept for this model. If set
        to False, no intercept will be used in calculations
        (e.g. data is expected to be already centered).
    replace : boolean, deafult False
        Whether or not to sample with replacement when "bootstrapping"
        in selection/estimation modules
    standardize : boolean, default False
        If True, the regressors X will be standardized before regression by
        subtracting the mean and dividing by their standard deviations. This
        parameter is equivalent to ``normalize`` in ``scikit-learn`` models.
    max_iter : int
        Maximum number of iterations for iterative fitting methods.
    random_state : int, RandomState instance, or None
        The seed of the pseudo random number generator that selects a random
        feature to update.  If int, random_state is the seed used by the random
        number generator; If RandomState instance, random_state is the random
        number generator; If None, the random number generator is the
        RandomState instance used by ``np.random``.
    comm : MPI communicator
        If passed, the selection and estimation steps are parallelized.
    logger : Logger
        The logger to use for messages when ``verbose=True`` in ``fit``.
        If *None* is passed, a logger that writes to ``sys.stdout`` will be
        used.
    penalty : string, 'l1' | scad' | 'mcp'
        Do selection with L1, SCAD or MCP regularization. Non-convex 
        regularization hyperparameter is fixed to a default constant in pyc.py

    Attributes
    ----------
    coef_ : nd-array, shape (n_features,) or (n_targets, n_features)
        Estimated coefficients for the linear regression problem.
    intercept_ : float
        Independent term in the linear model.
    supports_ : array, shape
        boolean array indicating whether a given regressor (column) is selected
        for estimation for a given regularization parameter value (row).
    """
    def __init__(self, n_boots_sel=24, n_boots_est=24, selection_frac=0.9,
                 estimation_frac=0.9, n_lambdas=100, stability_selection=1.,
                 estimation_score='BIC', estimation_target=None, eps=1e-3,
                 warm_start=True, copy_X=True, fit_intercept=True,
                 replace=False, standardize=True, max_iter=1000,
                 random_state=None, comm=None, logger=None,
                 penalty='scad'):
        super(UoI_NCV, self).__init__(
            n_boots_sel=n_boots_sel,
            n_boots_est=n_boots_est,
            selection_frac=selection_frac,
            estimation_frac=estimation_frac,
            estimation_target=estimation_target,
            stability_selection=stability_selection,
            copy_X=copy_X,
            fit_intercept=fit_intercept,
            replace=replace,
            standardize=standardize,
            random_state=random_state,
            comm=comm,
            estimation_score=estimation_score,
            max_iter=max_iter,
            logger=logger)
        self.n_lambdas = n_lambdas
        self.eps = eps
        if pycasso is None:
            raise ImportError('pycasso is not installed.')
        self._selection_lm = PycWrapper(
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            penalty=penalty)

        self._estimation_lm = OLS_Wrapper(fit_intercept=fit_intercept)

    def get_reg_params(self, X, y):
        alphas = _alpha_grid(
            X=X, y=y,
            l1_ratio=1.0,
            fit_intercept=self.fit_intercept,
            eps=self.eps,
            n_alphas=self.n_lambdas)

        return [{'alpha': a} for a in alphas]

    def uoi_selection_sweep(self, X, y, reg_param_values):
        """Overwrite base class selection sweep to accommodate pycasso
        path-wise solution"""

        alphas = np.array([reg_param['alpha']
                           for reg_param in reg_param_values])

        self._selection_lm.set_params(alphas=alphas)
        self._selection_lm.fit(X, y)

        return self._selection_lm.coef_
