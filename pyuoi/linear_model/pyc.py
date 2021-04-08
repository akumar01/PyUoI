import os
import numpy as np
from sklearn.model_selection import KFold
import pdb


try:
    import pycasso
except ImportError:
    pycasso = None

# Required for surpressing "Training is over" message in pycasso
class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in 
    Python, i.e. will suppress all print, even if the print originates in a 
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).      

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds =  [os.open(os.devnull,os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0],1)
        os.dup2(self.null_fds[1],2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0],1)
        os.dup2(self.save_fds[1],2)
        # Close all file descriptors
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


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

    def predict(self, X, fold=None):
        """Predicts responses given a design matrix.

        Parameters
        ----------
        X : ndarray, (n_samples, n_features)
            The design matrix.
        fold: int, optional
            If used as a cross-validated estimator, then predict using 
            the model associated with the fold integer
        Returns
        -------
        y : ndarray, shape (n_samples, n_reg_params)
            Predicted response vector for each regularization strength
        """
        if self.isfitted:
            if fold is not None:
                return np.matmul(X, self.coef_[fold, ...].T) + self.intercept_[fold, :]
            else:
                y = np.zeros((X.shape[0], self.coef_.shape[0]))
                for i in range(self.coef_.shape[0]):
                    y[:, i] = X @ self.coef_[i, :] + self.intercept_[i]
                return y
        else:
            raise NotFittedError('Estimator is not fit.')

    def fit(self, X, y, cross_validate=False):
        """Fit data according to the pycasso object.

        Parameters
        ----------
        X : ndarray, (n_samples, n_features)
            The design matrix.
        y : ndarray, shape (n_samples,)
            Response vector. Will be cast to X's dtype if necessary.
            Currently, this implementation does not handle multiple response
            variables.
        cross_validate : bool
            Whether or not we should fit the data across folds
        """
        if self.alphas is None:
            raise Exception('Set alphas before fitting.')

        # Sort in descending order
        self.alphas = np.sort(self.alphas)[::-1]

        if cross_validate:
            kfold =  KFold(n_splits=5)
            coefs = np.zeros((5, self.alphas.size, X.shape[1]))
            intercepts = np.zeros((5, self.alphas.size))
            test_idxs = []
            fold_idx = 0
            for train, test in kfold.split(X, y):
                Xtrain = X[train]
                ytrain = y[train]
                # No dummy path support for cross-validation
                solver = pycasso.Solver(Xtrain, ytrain, family='gaussian',
                                        useintercept=self.fit_intercept,
                                        lambdas=self.alphas,
                                        penalty=self.penalty,
                                        gamma=self.gamma,
                                        max_ite=self.max_iter)

                with suppress_stdout_stderr():
                    solver.train()
                coefs[fold_idx, ...] = solver.result['beta']
                intercepts[fold_idx, :] = solver.result['intercept']
                fold_idx += 1
                test_idxs.append(test)
            self.coef_ = coefs
            self.intercept_ = intercepts
            self.test_folds = test_idxs
        else:
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
            with suppress_stdout_stderr():
                self.solver.train()

            if dummy_path:
                self.coef_ = self.solver.result['beta'][0:pathlength, :]
                self.intercept_ = self.solver.result['intercept'][0:pathlength]
            else:
                self.coef_ = self.solver.result['beta']
                self.intercept_ = self.solver.result['intercept']

        self.isfitted = True
