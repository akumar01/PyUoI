import numpy as np
import sys
import logging
import pdb

def softmax(y, axis=-1):
    """Calculates the softmax distribution.

    Parameters
    ----------
    y : ndarray
        Log-probabilities.
    """

    yp = y - y.max(axis=axis, keepdims=True)
    epy = np.exp(yp)
    return epy / np.sum(epy, axis=axis, keepdims=True)


def sigmoid(x):
    """Calculates the bernoulli distribution.

    Parameters
    ----------
    x : ndarray
        Log-probabilities.
    """
    return np.exp(-np.logaddexp(0, -x))


def log_likelihood_glm(model, y_true, y_pred):
    """Calculates the log-likelihood of a generalized linear model given the
    true response variables and the "predicted" response variables. The
    "predicted" response variable varies by the specific generalized linear
    model under consideration.

    Parameters
    ----------
    model : string
        The generalized linear model to calculate the log-likelihood for.
    y_true : nd-array, shape (n_samples,)
        Array of true response values.
    y_pred : nd-array, shape (n_samples,)
        Array of predicted response values (conditional mean).

    Returns
    -------
    ll : float
        The log-likelihood.
    """

    # If y_true is of a different size than y_pred, trim away the beginning 
    # of y_true (used in autoregressive models)

    if y_true.size > y_pred.size:
        y_true = y_true[y_true.size - y_pred.size:]

    if model == 'normal':
        # this log-likelihood is calculated under the assumption that the
        # variance is the value that maximizes the log-likelihood
        rss = (y_true - y_pred)**2
        n_samples = y_true.size
        ll = -n_samples / 2 * (1 + np.log(np.mean(rss)))
    elif model == 'poisson':
        if not np.any(y_pred):
            if np.any(y_true):
                ll = -np.inf
            else:
                ll = 0.
        else:
            ll = np.mean(y_true * np.log(y_pred) - y_pred)
    else:
        raise ValueError('Model is not available.')
    return ll


def BIC(ll, n_features, n_samples):
    """Calculates the Bayesian Information Criterion.

    Parameters
    ----------
    ll : float
        The log-likelihood of the model.
    n_features : int
        The number of features used in the model.
    n_samples : int
        The number of samples in the dataset being tested.

    Returns
    -------
    BIC : float
        Bayesian Information Criterion
    """
    BIC = n_features * np.log(n_samples) - 2 * ll
    return BIC


def AIC(ll, n_features):
    """Calculates the Akaike Information Criterion.

    Parameters
    ----------
    ll : float
        The log-likelihood of the model.
    n_features : int
        The number of features used in the model.
    n_samples : int
        The number of samples in the dataset being tested.

    Returns
    -------
    AIC : float
        Akaike Information Criterion
    """

    AIC = 2 * n_features - 2 * ll
    return AIC


def AICc(ll, n_features, n_samples):
    """Calculate the corrected Akaike Information Criterion. This criterion is
    useful in cases when the number of samples is small.

    If the number of features is equal to the number of samples plus one, then
    the AIC is returned (the AICc is undefined in this case).

    Parameters
    ----------
    ll : float
        The log-likelihood of the model.
    n_features : int
        The number of features used in the model.
    n_samples : int
        The number of samples in the dataset being tested.

    Returns
    -------
    AIC : float
        Akaike Information Criterion
    """
    AICc = AIC(ll, n_features)
    if n_samples > (n_features + 1):
        AICc += 2 * (n_features**2 + n_features) / (n_samples - n_features - 1)
    return AICc


def check_logger(logger, name='uoi', comm=None):
    ret = logger
    if ret is None:
        if comm is not None and comm.Get_size() > 1:
            r, s = comm.Get_rank(), comm.Get_size()
            name += " " + str(r).rjust(int(np.log10(s)) + 1)

        ret = logging.getLogger(name=name)
        handler = logging.StreamHandler(sys.stdout)

        fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

        handler.setFormatter(logging.Formatter(fmt))
        ret.addHandler(handler)
    return ret

'''
    selection_accuracy
        beta: ndarray (n_models, n_features) or (n_features,)
        beta_hat : ndarray (n_models, n_features)
        threshold: Ignore magnitudes less than 1e-6
        sign_consistent: Assess whether correctly selected coefficients have the
        the right sign
'''
def selection_accuracy(beta, beta_hat, threshold = False, sign_consistent=False, var=False):

    beta, beta_hat = tile_beta(beta, beta_hat, var=var)

    if threshold:
        beta_hat[beta_hat < 1e-6] = 0

    selection_accuracy = np.zeros(beta_hat.shape[0])
    for i in range(beta_hat.shape[0]):
        b = beta[i, :].squeeze()
        bhat = beta_hat[i, :].squeeze()

        # Define support sets in terms of indices
        Sb = set(np.nonzero(b)[0].tolist())
        Sbhat = set(np.nonzero(bhat)[0].tolist())

        normalization = len(Sb) + len(Sbhat)
        # This will only occur if both sb and sbhat are all 0, in which 
        # case setting the normalization to 0 will give the intuitive result
        # (perfect selection accuracy)
        if normalization == 0:
            normalization = 1

        # If requiring sign consistency, need to define set intersection
        # in a sign consistent way
        if sign_consistent:
            selection_accuracy[i] = 1 - float(len(sign_consistent_set_diff(Sb, Sbhat, b, bhat)))\
                                    /float(normalization)            
        else:

            selection_accuracy[i] = 1 - \
            float(len((Sb.difference(Sbhat)).union(Sbhat.difference(Sb))))\
            /float(normalization)
    return selection_accuracy


def sign_consistent_set_diff(S, Shat, b, bhat):

    # Treat incorrect signs as false negatives

    # Treat incorrect signs as false negatives
    common_support = list(S.intersection(Shat))

    incorrect_signs = [idx for idx in common_support 
                           if np.sign(b[idx]) != np.sign(bhat[idx])]

    # Remove incorrect signs from Shat
    for idx in incorrect_signs:
        Shat.remove(idx)

    # Now take the symmetric set difference
    return (S.difference(Shat)).union(Shat.difference(S))

# Calculate estimation error
# Do so using only the overlap of the estimated and true support sets
def estimation_error(beta, beta_hat, threshold = False, var=False):
    beta, beta_hat = tile_beta(beta, beta_hat, var=var)

    if threshold:
        beta_hat[beta_hat < 1e-6] = 0

    ee = np.zeros(beta_hat.shape[0])
    median_ee = np.zeros(beta_hat.shape[0])

    for i in range(beta_hat.shape[0]):
        b = beta[i, :].squeeze()
        bhat = beta_hat[i, :].squeeze()

        common_support = np.bitwise_and(b != 0, bhat != 0)
        p = bhat[common_support].size
        if p > 0:
            median_ee[i] = np.median(np.sqrt(np.power(b[common_support] - \
                                        bhat[common_support], 2)))
            ee[i] = 1/p * np.sqrt(np.sum(np.power(b[common_support] - \
                                  bhat[common_support], 2)))
        else:
            median_ee[i] = np.nan
            ee[i] = np.nan

    return ee, median_ee


# Calculate the estimation error, separately measuring the contribution
# from selection mismatch (magnitude of false negatives + false positives)
# and estimatione rror (magnitude of error in correctly selected for coefficients)
def stratified_estimation_error(beta, beta_hat, threshold = False, var=False):
    beta, beta_hat = tile_beta(beta, beta_hat, var=var)

    if threshold:
        beta_hat[beta_hat < 1e-6] = 0

    fn_ee = np.zeros(beta_hat.shape[0])
    fp_ee = np.zeros(beta_hat.shape[0])
    estimation_ee = np.zeros(beta_hat.shape[0])

    for i in range(beta_hat.shape[0]):
        b = beta[i, :].squeeze()
        bhat = beta_hat[i, :].squeeze()

        common_support = np.bitwise_and(b != 0, bhat != 0)

        zerob = bhat[(b == 0)].ravel()
        false_positives = zerob[np.nonzero(zerob)]

        zerobhat = b[(bhat == 0).ravel()]
        false_negatives = zerobhat[np.nonzero(zerobhat)]
        fn_ee[i] = np.sum(np.abs(false_negatives))
        fp_ee[i] = np.sum(np.abs(false_positives))
        p = bhat[common_support].size
        if p > 0:
            estimation_ee[i] = np.sqrt(np.sum(np.power(b[common_support] - \
                                  bhat[common_support], 2)))
        else:
            estimation_ee[i] = 0

    return fn_ee, fp_ee, estimation_ee


def tile_beta(beta, beta_hat, var=False):

    if var:
        # Ravel VAR matrices into a vector for compatibility in estimation error
        # and selection accuracy checks
        beta = beta.reshape((-1, np.prod(beta.shape[-3:])))
        beta_hat = beta_hat.reshape((-1, np.prod(beta_hat.shape[-3:])))

    if np.ndim(beta_hat) == 1:
        beta_hat = beta_hat[np.newaxis, :]

    if np.ndim(beta) == 1:
        beta = beta[np.newaxis, :]

    if beta.shape != beta_hat.shape:
        beta = np.tile(beta, [int(beta_hat.shape[0]/beta.shape[0]), 1])

    return beta, beta_hat

