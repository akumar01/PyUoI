import numpy as np
from numpy.linalg import norm
import pdb
from scipy.optimize import minimize

from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.linear_model.base import _pre_fit
from sklearn.linear_model.coordinate_descent import _alpha_grid
from sklearn.utils import check_array, check_X_y

from .base import AbstractUoILinearRegressor

import matplotlib.pyplot as plt


class UoI_Spgrasso(AbstractUoILinearRegressor):

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





class SparseGroupLasso(ElasticNet):

	# max_iter1: how many times to iterate over groups
	# max_iter2: how many iterations to take in a given optimization within a given
	# group
	# Total number of iterations taken during optimization is therefore 
	# groups * max_iter1 * max_iter2
	def __init__(self, groups = None, alpha=1.0, l1_ratio=0.5, fit_intercept=True,
				 normalize=False, precompute=False, max_iter=1000,
				 copy_X=True, tol=1e-4, warm_start=False, positive=False,
				 random_state=None, selection='cyclic', max_iter1 = 1e2, max_iter2 =1e3):

		super(SparseGroupLasso, self).__init__(
		alpha=alpha, l1_ratio= l1_ratio, fit_intercept=fit_intercept,
		normalize=normalize, precompute=precompute, copy_X=copy_X,
		tol=tol, warm_start=warm_start,	positive=positive,
		random_state=random_state, selection=selection)

		self.groups = groups
		self.max_iter1 = int(max_iter1)
		self.max_iter2 = int(max_iter2)

	def spgrasso_penalty(self, beta, *args):
		# args: alpha, l1_ratio, X, y, groups
		alpha = args[0]
		l1_ratio = args[1]
		X = args[2]
		y = args[3]
		groups = args[4]
		n = X.shape[0]

		group_est = np.zeros(n)
		l2_group = 0
		for group in groups:
			group_est += np.dot(X[:, group], beta[group])
			l2_group += np.sqrt(len(group)) * norm(beta[group])**2

		return norm(y.ravel() - group_est)**2 + (1 - l1_ratio) * alpha * l2_group\
				+ l1_ratio * alpha * norm(beta, 1)

	def minimize(self, alpha, l1_ratio, X, y, beta_init = None):
		n, p = X.shape
		# Implement the algorithm described in the sparse group
		# lasso paper

		# IMPORTANT NOTES: alpha -> lambda in the paper, l1_ratio -> alpha
		# ALSO: Paper drops sqrt(p_k) terms from the getgo. We have to be careful
		# to modify all lambda (1 - alpha) terms from the paper's description of the
		# algorithm to read sqrt(p_k) lambda (1 - alpha)

		groups = self.groups

		# Functions needed for optimization

		# Soft thresholding operator:
		S = lambda x, y: np.sign(x) * np.maximum(abs(x) - y, np.zeros(x.shape))

		# Partial residual
		r = lambda X, y, beta, group: y - np.dot(X, beta)\
			+ np.dot(X[:, group], beta[group])

		# Loss function
		loss = lambda x, y: 1/(2 * x.shape[0]) * norm(x - y)**2

		# Loss function gradient
		grad_l = lambda X, r, group: -1* 1/X.shape[0] * X[:, group].T @ r

		# Update formula
		U = lambda X, r, beta_k, group, alpha, l1_ratio, t: \
		np.maximum(1 - np.sqrt(len(beta_k)) * (t * (1 - l1_ratio) * alpha)\
			/(norm(S(beta_k - t * grad_l(X, r, group), t * alpha * l1_ratio))), 0)\
			* S(beta_k - t * grad_l(X, r, group), t * alpha * l1_ratio)
		# Track solution change
		Delta = lambda x1, x2: x2 - x1

		# Track coefficient trajectories only for each sweep through the groups
		# to save space
		betas = np.zeros((p, self.max_iter2 + 1))

		# Initialize beta if provided
		if beta_init is not None:
			betas[:, 0] = beta_init
		else:
			# random initialization
			betas[:, 0] = np.random.uniform(-10, 10, p)

		k = 0

		while k < self.max_iter1:
			for group in groups:
				p_k = len(group)
				# Check if group coefficients are identically zero:
				if norm(S(X[:, group], alpha * l1_ratio)) <= np.sqrt(p_k) * (1 - l1_ratio) * alpha:
					betas[group, 0] = 0
					continue
				else:

					converged = 0
					# Step size
					t = 1
					# Iteration counter 
					l = 0

					theta_2 = betas[group, 0, np.newaxis]

					while (not converged) and (l < self.max_iter2 - 1):
						# Center of majorizing functions:
						beta = betas[:, l, np.newaxis]
						beta_k = beta[group]
						theta_1 = theta_2						
						# Update step size until majorization condition is satisfied:
						res = r(X, y, beta, group)
						U0 = U(X, res, beta_k, group, alpha, l1_ratio, t)

						while loss(res, X[:, group] @ U0) > loss(res, X[:, group] @ beta_k)\
						 + grad_l(X, res, group).T @ Delta(beta_k, U0) + 1/(2 * t) \
						 * norm(Delta(beta_k, U0))**2:

							t *= 0.8

						# Update theta:
						theta_2 = U(X, y, beta_k, group, alpha, l1_ratio, t)

						# Update betas:
						# NOTE: we have started with l = 0, so we incremenet l and use this value
						# in the update rule 

						# Increment l
						l += 1

						betas[group, l] = (theta_1 + (l)/(l + 3) * (theta_2 - theta_1)).ravel()

						# Check convergence:
						if norm(Delta(beta_k.ravel(), betas[group, l])) <= self.tol:
							converged = 1


						if l == self.max_iter2 - 1:
							print(norm(Delta(beta_k.ravel(), betas[group, l])))
					# Set the initial set of betas for the next run equal to the last set of betas
					# from this run:
					betas[group, 0] = betas[group, l]

			k += 1
		pdb.set_trace()
		return betas[:, 0]


	def fit(self, X, y):

		"""Fit model with coordinate descent.
		Parameters
		-----------
		X : ndarray or scipy.sparse matrix, (n_samples, n_features)
			Data
		y : ndarray, shape (n_samples,) or (n_samples, n_targets)
			Target. Will be cast to X's dtype if necessary
		check_input : boolean, (default=True)
			Allow to bypass several input checking.
			Don't use this parameter unless you know what you do.
		Notes
		-----
		Coordinate descent is an algorithm that considers each column of
		data at a time hence it will automatically convert the X input
		as a Fortran-contiguous numpy array if necessary.
		To avoid memory re-allocation it is advised to allocate the
		initial data in memory directly using that format.
		"""

		if self.alpha == 0:
			warnings.warn("With alpha=0, this algorithm does not converge "
						  "well. You are advised to use the LinearRegression "
						  "estimator", stacklevel=2)

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

		if self.groups is None:
			self.groups = np.split(np.arange(n_features), n_features)

		for k in range(n_targets):
			if Xy is not None:
				this_Xy = Xy[:, k]
			else:
				this_Xy = None

			# result = minimize(self.spgrasso_penalty, np.random.uniform(-10, 10, n_features),
			# 	args=(self.alpha, self.l1_ratio, X, y, self.groups), method='BFGS',
			# 	tol=self.tol)            

			# coef_[k] = result.x

			coef_[k] = self.minimize(self.alpha, self.l1_ratio, X, y)

		if n_targets == 1:
			self.coef_ = coef_[0]
		else:
			self.coef_ = coef_

		self._set_intercept(X_offset, y_offset, X_scale)

		# workaround since _set_intercept will cast self.coef_ into X.dtype
		self.coef_ = np.asarray(self.coef_, dtype=X.dtype)

		# return self for chaining fit and predict calls
		return self
