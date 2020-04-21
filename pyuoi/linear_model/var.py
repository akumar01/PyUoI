import numpy as np 
from .base import AbstractUoILinearRegressor
from .ncvr import UoI_NCVR


class VAR(UoI_NCV, AbstractUoILinearRegressor):
    r"""UoI\ :sub:`VAR` solver.

    Parameters
    order: int
        The range of lags to do regression over

    penalty : string, 'l1' | 'scad' | 'mcp'
        Use either Lasso, SCAD, or MCP for selection

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
    def __init__(self, order=1, penalty='l1', **uoi_kwargs):
        self.order = order
        super(VAR, self).__init__(penalty=penalty **uoi_kwargs)

    def _resample(self, idxs, sampling_frac, stratify):
        """Modify default resampling behavior to use moving block 
        boot strapping"""

        return resample('block', idxs, self.replace, self.random_state,
                        sampling_frac=sampling_frac, stratify=stratify)

    def form_VAR(y):
        n_samples, n_features = y.shape

        # Preallocate
        X = np.zeros((n_samples - self.order, n_features, self.order))
        
        for i in range(self.order, n_samples):

            X[i - self.order, :, :] = np.array([y[i - j, :] 
                                                for j in range(self.order)])

        # Reshape
        X = np.reshape(X, (n_samples, -1))


    def fit(y):
        """
            Overall default UoI fit
            y: ndarray of shape (n_samples, n_dof)
        """
        X, y = self.form_VAR(y)

        # Feed into normal fit. Will have try different normalization
        # strategies eventually
        super(VAR, self).fit(X, y)
