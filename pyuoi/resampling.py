import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state


def resample(type, x, replace=True, random_state=None, **kwargs):
    r"""Takes the data in X and y and returns the appropriate
    resampled versions.

	Parameters
    ----------
    type : str "bootstrap" | "block"
        type of resampling to do
    x : array-like with shape (n_samples,)
        A suitable array whose indices will be used as the basis for resampling.
    replace : bool
        Should we sample with replacement or not? True corresponds to a
        bootstrap, False to use sklearn train_test_split
    random_state : int or np.random.RandomState instance
        If an int or None, used to create a np.random.RandomState instance.
        Else, if already an instance of RandomState, used as-is to generate
        random samples
    **kwargs : arguments necessary to perform the desired resampling.
    See functions below for necessary keyword arguments

	Returns
    -------
	train_idxs: ndarray, shape (n_train_samples, n_features)
	test_idxs: ndarray, shape (n_test_samples, n_features)
    """
    random_state = check_random_state(random_state)

    if type == 'bootstrap':
        train_idxs, test_idxs = bootstrap(x, random_state, replace, **kwargs)
    elif type == 'block':
        train_idxs = block_bootstrap(x, random_state, **kwargs)
        # Return dummy test_idxs for backwards compatibility
        test_idxs = train_idxs

    return train_idxs, test_idxs


def bootstrap(x, rand_state, replace, sampling_frac=0.9, stratify=None):
    r"""Sample with or without replacement. For test idxs, we take the 
    complement of the unique boostrap indices to ensure there is no overlap. 
    This implies that test_frac does not necessarily equal 1 - train_frac

	Parameters
	----------
	rand_state : np.random.RandomState instance
		Use for random index selection
	replace : bool
		Sample with or without replacement
    sampling_frac : float between 0 and 1
        What fraction of x to allocate to test data
    stratify : array-like
        Class labels for stratified bootstrapping

	Returns
	-------
	train_idxs: ndarray, shape (n_train_samples, n_features)
	test_idxs: ndarray, shape (n_test_samples, n_features)
    """
    if replace:

        if stratify is not None:
            idxs = np.arange(len(x))
            train_idxs = []
            test_idxs = []

            for class_, class_size in Counter(stratify).items():
                # For each class, sample proportional to its membership
                n_samples = int(class_size * sampling_frac)
                class_idxs = idxs[stratify == class_]

                class_train_idxs = rand_state.choice(class_idxs,
                                                     size=n_samples,
                                                     replace=True)
                class_test_idxs = np.setdiff1d(class_idxs, class_train_idxs)

                train_idxs.extend(class_train_idxs)
                test_idxs.extend(class_test_idxs)
        else:
            n_samples = int(len(x) * sampling_frac)

            train_idxs = rand_state.choice(len(x), size=n_samples,
                                           replace=True)
            test_idxs = np.setdiff1d(np.arange(len(x)), train_idxs)

    else:

        train_idxs, test_idxs = train_test_split(x, train_size=sampling_frac,
                                                 test_size=1 - sampling_frac,
                                                 random_state=rand_state,
                                                 stratify=stratify)

    return train_idxs, test_idxs


def block_bootstrap(x, rand_state, L):
    r"""Use a moving block bootstrap for resampling in VAR models. See
	'The jackknife and the bootstrap for general stationary observations'
	by Hans Kunsch, Annals of Statistics 1989, Vol 17, No. 3, 1217-1241		

	Parameters
	----------
	rand_state : np.random.RandomState instance
		Use for random index selection
	L : int
		Size of bootstrap blocks
    block_frac : float between 0 and 1
        What fraction of blocks to allocate to test data

	Returns
	-------
	train_idxs: ndarray, shape (n_train_samples, n_features)
    """    
    n = len(x)

    # Divide the data set into overlapping blocks of length L
    blocks = []
    for i in range(x.size - L + 1):
        blocks.append(x[i:i + L])

    n_sampled_blocks = int(x.size/L)
    selected_blocks = rand_state.choice(len(blocks),
                                        size=n_sampled_blocks,
                                        replace=True)

    # Stitch together the blocks
    train_idxs = []
    for idx in selected_blocks:
        train_idxs.extend(blocks[idx])

    return train_idxs
