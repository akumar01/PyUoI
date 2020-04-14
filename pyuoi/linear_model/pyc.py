import numpy as np

try:
    import pycasso
except ImportError:
    pycasso = None

# default values for non-convex penalty term 
gamma = {'l1': 0., 'scad': 3.7, 'mcp': 3.}

class PycWrapper():
    """Sklearn syntax wrapper for seamless integration of solvers based
    on the pycasso package.

    Parameters
    ----------
    penalty:'l1', 'scad', or 'mcp' 
    alphas : nd-array
        The regularization path. Defaults to None for compatibility with UoI,
        but needs to be set prior to fitting.
    fit_intercept : bool
        Whether to calculate the intercept for this model. If set to ``False``,
        no intercept will be used in calculations.
    max_iter : int
        Maximum number of iterations for pycasso solver.

    Attributes
    ----------
    coef_ : ndarray, shape (n_features,) or (n_targets, n_features)
        Estimated coefficients for the linear regression problem.
    intercept_ : float
        Independent term in the linear model.
    """
    def __init__(self, alphas=None, fit_intercept=False, max_iter=1000,
    			 penalty='l1'):
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.alphas = alphas
        # Flag to prevent us from predicting before fitting
        self.isfitted = False

        self.penalty = penalty
        self.gamma = gamma[penalty]

    def set_params(self, **kwargs):
        """Sets the parameters of this estimator."""
        _valid_params = ['alphas', 'fit_intercept', 'max_iter']

        for key, value in kwargs.items():
            if key in _valid_params:
                setattr(self, key, value)
            else:
                raise ValueError('Invalid parameter %s' % key)

    def predict(self, X):
        """Predicts responses given a design matrix.

        Parameters
        ----------
        X : ndarray, (n_samples, n_features)
            The design matrix.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Predicted response vector.
        """
        if self.isfitted:
            return np.matmul(X, self.coef_.T) + self.intercept_
        else:
            raise NotFittedError('Estimator is not fit.')

    def fit(self, X, y):
        """Fit data according to the pycasso object.

        Parameters
        ----------
        X : ndarray, (n_samples, n_features)
            The design matrix.
        y : ndarray, shape (n_samples,)
            Response vector. Will be cast to X's dtype if necessary.
            Currently, this implementation does not handle multiple response
            variables.
        """
        if self.alphas is None:
            raise Exception('Set alphas before fitting.')

        # Sort in descending order
        self.alphas = np.sort(self.alphas)[::-1]

        # Pycasso requires the regularization path to include at
        # least 3 regularization parameters.
        dummy_path = False
        if self.alphas.size < 3:
            dummy_path = True
            alphas = list(self.alphas)
            pathlength = len(alphas)
            while len(alphas) < 3:
                alphas.append(alphas[-1] / 2)
            self.alphas = np.array(alphas)

        self.solver = pycasso.Solver(X, y, family='gaussian',
                                     useintercept=self.fit_intercept,
                                     lambdas=self.alphas,
                                     penalty=self.penalty,
                                     gamma=self.gamma,
                                     max_ite=self.max_iter)
        self.solver.train()

        if dummy_path:
            self.coef_ = self.solver.result['beta'][0:pathlength, :]
            self.intercept_ = self.solver.result['intercept'][0:pathlength]
        else:
            self.coef_ = self.solver.result['beta']
            self.intercept_ = self.solver.result['intercept']

        self.isfitted = True
