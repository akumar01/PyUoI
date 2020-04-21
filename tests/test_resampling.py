import numpy as np
import scipy
import pytest
import pdb
import time
from sklearn.model_selection import train_test_split
from pyuoi.resampling import resample


def empirical_CDF(X, min_=-3, max_=3, nbins=50):
	"""Utility function to calculate empirical CDF and inverse from
	CDF from data X"""
	edges = np.linspace(min_, max_, nbins)
	F = np.zeros(nbins)
	n = X.size
	for i, edge in enumerate(edges):
		F[i] = 1/n * X[X >= edge].size

	# Inverse CDF
	Finv = scipy.interpolate.interp1d(F, edges)

	return F, Finv


def test_resample():
	"""Test that the resample function without replacement
	behaves identically to sklearn's train_test_split"""
	X = np.arange(10) @ np.eye(10)	
	train1, test1 = sklearn.train_test_split(X, train_size=0.9, test_size=0.1,
											  random_state=1234)
	train2, test2 = resample('bootstrap', X, replace=False, sampling_frac=0.9,
							 random_state=1234)

	assert(train1 == train2)
	assert(test1 == test2)


def test_bootstrap():
	"""Test that the bootstrapped mean is normally
	distributed about the empirical mean, a fundamental 
	property of the bootstrap technique. Also test for 
	duplicates"""

	n = 500
	B = 50
	np.random.seed(1234)

	# Variance 1 uniform distribution
	X = np.random.uniform(0, np.sqrt(12), size=n)

	T = np.zeros(B)
	repeats = False
	for b in range(B):
		bidxs, _ = resample('bootstrap', X, replace=True,
							sampling_frac=0.9)
		Xb = X[bidxs]

		if np.unique(Xb).size < Xb.size:
			repeats = True

		T[b] = np.sqrt(Xb.size) * (np.mean(Xb) - np.mean(X))

	D, p = scipy.stats.kstest(T, 'norm')
	
	# KS test p value is greater than 98 %
	assert(p > 0.98)
	# Should be repeats somewhere
	assert(repeats)


def test_block_boostrap():
	"""Test that the block bootstrap contains duplicates and that it does 
	a good job measuring the bias of OLS estimates of an AR(1) model with
	small sample sizes"""
	
	# Generate data from an AR(1) model
	np.random.seed(12345)

	beta = np.random.uniform(-1, 1)

	# Initial conditions
	n = 25
	y = np.zeros(n)
	y[0] = np.random.uniform()

	# Generate the data with additive noise and a constant offset
	c = 2

	# True mean:
	mu = c/(1 - beta)

	mu_hat = 

	# Also calculate bootstrap estimates and measure how good the bias 
	# correction is
	for rep in range(1000):
		for i in range(1, n):
			y[i] = y[i - 1] * beta + \
					  np.random.normal(scale=0.25) + c


	# Sample mean:
	y = np.mean(y)




if __name__ == '__main__':
	test_bootstrap()