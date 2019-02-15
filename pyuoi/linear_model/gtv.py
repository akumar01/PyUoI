import numpy as np
from numpy.linalg import norm
import pdb
from scipy.optimize import minimize
import quadprog

from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.linear_model.base import _pre_fit
from sklearn.linear_model.coordinate_descent import _alpha_grid
from sklearn.utils import check_array, check_X_y

from .base import AbstractUoILinearRegressor



class UoI_GTV(AbstractUoILinearRegressor):

    def __init__(self, groups = None, n_lambdas=48, alphas=np.array([0.5]),
                 n_boots_sel=48, n_boots_est=48, selection_frac=0.9,
                 estimation_frac=0.9, stability_selection=1.,
                 estimation_score='r2', warm_start=True, eps=1e-3,
                 copy_X=True, fit_intercept=True, normalize=True,
                 random_state=None, max_iter=1000,
                 comm=None):
        super(UoI_Spgrasso, self).__init__(
            n_boots_sel=n_boots_sel,
            n_boots_est=n_boots_est,
            selection_frac=selection_frac,
            estimation_frac=estimation_frac,
            stability_selection=stability_selection,
            estimation_score=estimation_score,
            copy_X=copy_X,
            fit_intercept=fit_intercept,
            normalize=normalize,
            random_state=random_state,
            comm=comm
        )
        self.n_lambdas = n_lambdas
        self.alphas = alphas
        self.n_alphas = len(alphas)
        self.warm_start = warm_start
        self.eps = eps
        self.lambdas = None
        self.__selection_lm = SparseGroupLasso(
            groups = groups,
            fit_intercept=fit_intercept,
            normalize=normalize,
            max_iter=max_iter,
            copy_X=copy_X,
            warm_start=warm_start,
            random_state=random_state)
        self.__estimation_lm = LinearRegression()

    @property
    def estimation_lm(self):
        return self.__estimation_lm

    @property
    def selection_lm(self):
        return self.__selection_lm

    def get_reg_params(self, X, y):
        """Calculates the regularization parameters (alpha and lambda) to be
        used for the provided data.

        Note that the Elastic Net penalty is given by

                1 / (2 * n_samples) * ||y - Xb||^2_2
            + lambda * (alpha * |b|_1 + 0.5 * (1 - alpha) * |b|^2_2)

        where lambda and alpha are regularization parameters.

        Note that scikit-learn does not use these names. Instead, scitkit-learn
        denotes alpha by 'l1_ratio' and lambda by 'alpha'.

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
            (lambda, alpha) describing the type of regularization to impose.
            The keys adhere to scikit-learn's terminology (lambda->alpha,
            alpha->l1_ratio). This allows easy passing into the ElasticNet
            object.
        """
        if self.lambdas is None:
            self.lambdas = np.zeros((self.n_alphas, self.n_lambdas))
            # a set of lambdas are generated for each alpha value (l1_ratio in
            # sci-kit learn parlance)
            for alpha_idx, alpha in enumerate(self.alphas):
                self.lambdas[alpha_idx, :] = _alpha_grid(
                    X=X, y=y,
                    l1_ratio=alpha,
                    fit_intercept=self.fit_intercept,
                    eps=self.eps,
                    n_alphas=self.n_lambdas,
                    normalize=self.normalize)

        # place the regularization parameters into a list of dictionaries
        reg_params = list()
        for alpha_idx, alpha in enumerate(self.alphas):
            for lamb_idx, lamb in enumerate(self.lambdas[alpha_idx]):
                # reset the regularization parameter
                reg_params.append(dict(alpha=lamb, l1_ratio=alpha))

        return reg_params





class GraphTotalVariance(ElasticNet):

    # max_iter1: how many times to iterate over groups
    # max_iter2: how many iterations to take in a given optimization within a given
    # group
    # Total number of iterations taken during optimization is therefore 
    # groups * max_iter1 * max_iter2
    def __init__(self, lambda_S, lambda_TV, lambda_1, fit_intercept=True,
                 normalize=False, precompute=False, max_iter=1000,
                 copy_X=True, tol=1e-4, warm_start=False, positive=False,
                 random_state=None, selection='cyclic'):

        super(GraphTotalVariance, self).__init__(
        fit_intercept=fit_intercept,
        normalize=normalize, precompute=precompute, copy_X=copy_X,
        tol=tol, warm_start=warm_start, positive=positive,
        random_state=random_state, selection=selection)

        self.lambda_S = lambda_S
        self.lambda_TV = lambda_TV
        self.lambda_1 = lambda_1


    # Transform the GTV objective into a quadratic programming problem
    def gtv_quadprog(self, *args):
        # args: lambda_S, lambda_TV, lambda_1, X, y, cov
        lambda_S = args[0]
        lambda_TV = args[1]
        lambda_1 = args[2]
        X = args[3]
        y = args[4]
        cov = args[5]
        n = X.shape[0]
        p = X.shape[1]

        # Assemble edge set from the covariance matrix:
        E = []
        for i in range(p):
            for j in range(p):
                if i == j: 
                    continue
                if cov[i, j] != 0:
                    E.append([i, j])

        # Coordinate transformations:   
        Gamma = np.zeros((len(E), p))

        for i in range(Gamma.shape[0]):
            e_jl = np.zeros(p)
            e_kl = np.zeros(p)
            e_jl[E[i][0]] = 1
            e_kl[E[i][1]] = 1
            Gamma[i, :] = np.sqrt(cov[E[i][0], E[i][1]]) * (e_jl - np.sign(cov[E[i][0], E[i][1]]) * e_kl)

        # Check shape of these!
        XX = np.concatenate([X, np.sqrt(n * lambda_S) * Gamma])
        YY = np.concatenate([y, np.zeros((len(E), 1))])
        GG = np.concatenate([lambda_TV * Gamma, np.identity(p)])

        # Constraints
        # t is inversely proportional to lambda_1
        t = 0.1/lambda_1          

        # Break the beta coefficients into beta_+ and beta_-

        # There are constraints that each vector of these be greater than 0 (need to check if we have to
        # feed in that the negative of these coefficients be less than 0)
        # Further constraint: the sum of beta_+ and beta_- must be less than or equal to t

        # Inequality constraint matrix:
        A = np.concatenate([np.ones((1, p)) , -1* np.ones((1, p))], axis = 1)
        A = np.concatenate([A, -1*np.identity(2 * p)])

        # Inequality constraint vector:
        h = np.concatenate([np.array([t]), np.zeros(2*p)])
        # Quadratic programming objective function
        Q =  XX.T @ XX
        c =  -XX.T @ YY

        # Enlarge the dimension of Q to handle the positive/negative decomposition
        QQ = np.concatenate([Q, -Q], axis = 1)
        QQ = np.concatenate([QQ, -QQ])

        cc = np.concatenate([c, -c])

        return QQ, cc, A, h


    # Test to see whether we can make ordinary lasso work with quadratic programming
    def lasso_quadprog(self, *args):
        lambda1 = args[0]
        X = args[1]
        y = args[2]

        n = X.shape[0]
        p = X.shape[1]


        t = 1/lambda1         

        # Constraints
        # Inequality constraint matrix:
        A = np.concatenate([np.ones((1, p)) , -1* np.ones((1, p))], axis = 1)
        A = np.concatenate([A, -1*np.identity(2 * p)])

    
        # Inequality constraint vector:
        h = np.concatenate([np.array([t]), np.zeros(2*p)])

        Q = X.T @ X
        c = -X.T @ y

        # Enlarge the dimension of Q to handle the positive/negative decomposition
        QQ = np.concatenate([Q, -Q], axis = 1)
        QQ = np.concatenate([QQ, -QQ])

        cc = np.concatenate([c, -c])


        return QQ, cc, A, h


    def minimize(self, lambda_S, lambda_TV, lambda_1, X, y, cov):
        # use quadratic programming to optimize the GTV loss function

#        Q, c, A, h = self.gtv_quadprog(lambda_S, lambda_TV, lambda_1, X, y, cov)
        Q, c, A, h = self.lasso_quadprog(lambda_1, X, y)

        # Need a symmetric, positive definite matrix for solvers 
        Q = 1/2 * (Q + Q.T + np.identity(Q.shape[0]) * 1e-4)

        # Negate the sign to match the standard form of the solve 
        c = -c.ravel()

        # Quadprog has >= inequality constraints whereas we formulate as <= constraints
        A = -A.T

        h = -h

        solution = quadprog.solve_qp(Q, c, A, h)

        # Recover actual coefficients
        coeffs_pm = solution[0]
        coeffs = coeffs_pm[0:int(len(coeffs_pm)/2)] - coeffs_pm[int(len(coeffs_pm)/2)::]

        return coeffs

    def fit(self, X, y, cov):

        """Fit model with coordinate descent.
        Parameters
        -----------
        X : ndarray or scipy.sparse matrix, (n_samples, n_features)
            Data
        y : ndarray, shape (n_samples,) or (n_samples, n_targets)
            Target. Will be cast to X's dtype if necessary

        cov : Estimated data covariance matrix

        Notes
        -----
        Coordinate descent is an algorithm that considers each column of
        data at a time hence it will automatically convert the X input
        as a Fortran-contiguous numpy array if necessary.
        To avoid memory re-allocation it is advised to allocate the
        initial data in memory directly using that format.
        """

        # Remember if X is copied
        X_copied = False
        X_copied = self.copy_X and self.fit_intercept
        X, y = check_X_y(X, y, accept_sparse='csc',
                         order='F', dtype=[np.float64, np.float32],
                         copy=X_copied, multi_output=True, y_numeric=True)
        y = check_array(y, order='F', copy=False, dtype=X.dtype.type,
                        ensure_2d=False)

        # Ensure copying happens only once, don't do it again if done above
        should_copy = self.copy_X and not X_copied
        X, y, X_offset, y_offset, X_scale, precompute, Xy = \
            _pre_fit(X, y, None, self.precompute, self.normalize,
                     self.fit_intercept, copy=should_copy,
                     check_input=True)
        if y.ndim == 1:
            y = y[:, np.newaxis]
        if Xy is not None and Xy.ndim == 1:
            Xy = Xy[:, np.newaxis]

        n_samples, n_features = X.shape
        n_targets = y.shape[1]

        if self.selection not in ['cyclic', 'random']:
            raise ValueError("selection should be either random or cyclic.")

        if not self.warm_start or not hasattr(self, "coef_"):
            coef_ = np.zeros((n_targets, n_features), dtype=X.dtype,
                             order='F')
        else:
            coef_ = self.coef_
            if coef_.ndim == 1:
                coef_ = coef_[np.newaxis, :]

        for k in range(n_targets):
            if Xy is not None:
                this_Xy = Xy[:, k]
            else:
                this_Xy = None

            coef_[k] = self.minimize(self.lambda_S, self.lambda_TV, self.lambda_1, X, y, cov)

        if n_targets == 1:
            self.coef_ = coef_[0]
        else:
            self.coef_ = coef_

        self._set_intercept(X_offset, y_offset, X_scale)

        # workaround since _set_intercept will cast self.coef_ into X.dtype
        self.coef_ = np.asarray(self.coef_, dtype=X.dtype)

        # return self for chaining fit and predict calls
        return self
