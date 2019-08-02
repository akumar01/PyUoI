import numpy as np
from collections import Counter

from sklearn.model_selection import train_test_split
def resample(type, X, y, **kwargs):
	''' function resample: Takes the data in X and y and returns the appropriate 
	resampled versions.

	type : str "train_test_split" | "bootstrap" | "block train_test_split" | block bootstrap"
		type of resampling to do
		
	X : array-like with shape (n_samples, n_features)
	y : array-like with shape (n_samples,)

	**kwargs : arguments necessary to perform the desired resampling. See functions 
	below for necessary keyword arguments'''

	if type == 'train_test_split':
		train_idxs, test_idxs = train_test_split(np.arange(X.shape[0]),
				         						 test_size=1 - sampling_frac,
				         						 stratify=stratify,
				         						 random_state=random_state)
	elif type == 'bootstrap':
		train_ids, test_idxs = bootstrap(X, y, sampling_frac = sampling_frac, 
										 stratify=stratify, random_state=random_state)		

	return [train_idxs, test_idxs]

def bootstrap(X, y, sampling_frac, stratify=None, random_state=None):
	'''Sample with replacement. For test idxs, we take the complement 
	of the unique boostrap indices to ensure there is no overlap. This
	implies that test_frac does not necessarily equal 1 - train_frac'''

	if random_state is not None:
		np.random.seed(random_state)

	if stratify is not None:
		idxs = np.arange(X.shape[0])
		train_idxs = []
		test_idxs = []

		for class_, class_size in Counter(stratify):
			# For each class, sample proportional to its membership
			n_samples = int(class_size * sampling_frac)
			class_idxs = idxs[stratify==class_] 
			class_train_idxs = np.random.choice(class_idxs, 
							                    size = n_samples, 
							                    replace = True)
			class_test_idxs = 
			list(set(class_idxs).difference(set(class_train_idxs)))

			train_idxs.extend(class_train_idxs)
			test_idxs.extend(class_test_idxs)

	else:
		n_samples = int(X.shape[0] * sampling_frac)

		train_idxs = np.random.choice(np.arange(X.shape[0]), 
									  size = n_samples, 
									  replace=True)

		test_idxs = list(set(np.arange(X.shape[0])).difference(set(train_idxs)))

	return train_idxs, test_idxs

def block_train_test_split(X, y, sampling_frac, L, random_state=None):
	'''Moving block bootstrap. Divide the time series of length n_samples
	into overlapping blocks of length L. Then, choose one of those blocks
	and sample, without replacement, 

	if random_state is not None:
		np.random.seed(random_state)




def block_boostrap():
	pass