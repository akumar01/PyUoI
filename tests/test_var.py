import pytest 
import numpy as np
from scipy.stats import special_ortho_group
from pyuoi.linear_model.var import VAR, form_lag_matrix

import pdb

# Tests to do:
# Test problem assembly
# Test selection and fitting in easy problem

# Make a toy VAR model
def make_var_model(n_samples, n_features, order, seed, noise=True):

	np.random.seed(seed)
	X = np.zeros((n_samples, n_features))

	# initial condition 
	X[:order, :] = np.random.normal(size=(order, n_features))

	# Coefficients. VAR stability condition guaranteed if we 
	# sample between -1 and 1
	A = []
	for j in range(order):
		eigvals = np.random.uniform(-1, 1, size=(n_features,))
		# Rotate randomly
		S = special_ortho_group.rvs(n_features)
		A_j = S @ np.diag(eigvals) @ np.linalg.inv(S)
		A.append(A_j)

	for i in range(order, n_samples):

		X[i, :] = np.sum(np.array([X[i - j, :] @ A[j] for j in range(order)]),
		 				 axis=0)
		if noise:
			X[i, :] += np.random.normal(scale=0.1, size=(n_features,))

	return X, A

# def test_form_lag_matrix():
# 	seed = 1234
# 	n_samples = 100
# 	n_features = 10
# 	order = 2
# 	X, _ = make_var_model(n_samples, n_features, order, seed)

# 	X, y = form_lag_matrix(X, X[:, 0], order)

# 	assert(X.shape == (n_samples - order, n_features * order))
# 	assert(y.size == n_samples - order)


def test_uoi_var():

	# Test on a problem with no noise

	# Sparsify coefficients! Just invent a toy problem that you 
	# know the answer to
	X, A = make_var_model(100, 3, 2, seed=1234, noise=False)

	var = UoI_VAR(order=2, random_state=123, penalty='l1')
 
	var.fit(X)

	pdb.set_trace()
