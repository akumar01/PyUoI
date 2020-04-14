import numpy as np 
from .lasso import UoI_Lasso
from .ncvr import UoI_NCVR

class VAR():
    r"""UoI\ :sub:`Lasso` solver.

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
    solver : string, 'cd' | 'pyc'
        If cd, will use the ``scikit-learn`` lasso implementation (via
        coordinate descent). If pyc, will use pyclasso, built off of the
        pycasso path-wise solver.


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
    def __init__(self, estimator='l1', random_state=None,
                 comm=None, **uoi_kwargs):

    	if estimator == 'l1':
    		self.estimator = UoILasso(comm=comm, random_state=random_state,
    								  **uoi_kwargs)
    	else:
    		self.estimator = UoI_NCVR(comm=comm, random_state=random_state,
    								  **uoi_kwargs)

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

        if self.solver == 'pyc':
            alphas = np.array([reg_param['alpha']
                               for reg_param in reg_param_values])

            self._selection_lm.set_params(alphas=alphas)
            self._selection_lm.fit(X, y)

            return self._selection_lm.coef_
        else:
            return super(UoI_Lasso, self).uoi_selection_sweep(X, y,
                                                              reg_param_values)
