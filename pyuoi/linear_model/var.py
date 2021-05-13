import numpy as np 
from scipy.signal import convolve
from sklearn.utils import check_X_y, check_random_state
from sklearn.linear_model import LinearRegression
from sklearn.linear_model._coordinate_descent import _alpha_grid
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from pyuoi.resampling import resample
from pyuoi.utils import log_likelihood_glm, BIC
from .base import AbstractUoILinearRegressor
from .ncv import UoI_NCV
from .pyc import PycWrapper

from mpi4py import MPI
from pyuoi.mpi_utils import Gatherv_rows

from numpy.lib.stride_tricks import as_strided

import pdb
import time
import os
import pickle

# Tiered communicators
def comm_setup(comm, ncomms):

    if comm is not None:
        rank = comm.rank
        numproc = comm.Get_size()

        ranks = np.arange(numproc)
        split_ranks = np.array_split(ranks, ncomms)
        color = [i for i in np.arange(ncomms) if rank in split_ranks[i]][0]
        subcomm_roots = [split_ranks[i][0] for i in np.arange(ncomms)]

        subcomm = comm.Split(color, rank)

        # Setup a root communicator that links the roots of each subcomm
        global_group = comm.Get_group()
        root_group = MPI.Group.Incl(global_group, subcomm_roots)
        print('rank: %d, color %d, subcomm_rank: %d, root_group_rank: %d' % (rank, color, subcomm.rank, root_group.rank))
        roots_comm = comm.Create(root_group)


    else:
        subcomm = None
        roots_comm = None
        color = None
        savepaths = None
        root_group = None

    return comm, subcomm, roots_comm, color, root_group


class VAR():
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
    def __init__(self, order=1, random_state=None, penalty='l1', 
                 estimator='uoi', self_regress=False, 
                 comm=None, ncomms=1, **estimator_kwargs):
        self.order = order
        self.self_regress = self_regress
        self.random_state = check_random_state(random_state)
        self.comm = comm
        self.ncomms = ncomms

        comm, subcomm, rootcomm, color, root_group = comm_setup(self.comm, self.ncomms)

        self.subcomm = subcomm
        self.rootcomm = rootcomm
        self.root_group = root_group
        self.color = color

        if estimator == 'uoi':
            self.estimator = UoIVAR_Estimator(penalty=penalty, 
                                              random_state=self.random_state, 
                                              comm = self.subcomm,
                                              **estimator_kwargs)

        elif estimator == 'ncv':
            self.estimator = NCV_VAR_Estimator(penalty=penalty,
                                               random_state=self.random_state,
                                               **estimator_kwargs)
        elif estimator == 'ols':
            self.estimator = VAR_OLS_Wrapper(standalone=True,
                                             normalize=True,
                                             **estimator_kwargs)
        else: 
            raise ValueError('Unknown estimator')

    def fit(self, y, distributed_save=False, savepath=None):
        """
            y: ndarray of shape (n_samples, n_dof)
               or (n_trials, n_samples, n_dof)
        """

        if y.ndim == 2:
            n_samples, n_dof = y.shape
        elif y.ndim == 3:
            n_trials, n_samples, n_dof = y.shape
        else:
            raise ValueError('Shape of y must be either (n_samples, n_dof) or (n_trials, n_samples, n_dof)')

        # Setup directory for distributed save
        if self.comm is not None and distributed_save:
            if self.comm.rank == 0:
                print('Initializing save directory')
                if not os.path.exists(savepath):
                    os.makedirs(savepath)
        elif distributed_save:
            if not os.path.exists(savepath):
                os.makedirs(savepath)

        savepaths = []
        for i in range(n_dof):
            savepaths.append('%s/%s.dat' % (savepath, i))

        # Regress each column (feature) against all the others

        # Spread each row across mpi tasks. If ncomms is > 1, we also spread
        # estimation of each row
        if self.comm is not None:
            task_list = np.array_split(np.arange(n_dof), self.ncomms)[self.color]
            num_tasks = len(task_list)

            scores = []
            coefs = []

            intercept = np.zeros(num_tasks)
            # XX, YY = _form_var_problem(y, self.order, self.self_regress)

            for idx, i in enumerate(task_list):
                print('Rank %d working on task %d' % (self.comm.rank, i))
                xx, yy = _form_var_problem(y, i, self.order, self.self_regress)

                if distributed_save:
                    self.estimator.fit(xx, yy, savepaths[i])
                else:
                    self.estimator.fit(xx, yy)

                if self.self_regress:
                    coefs_ = np.reshape(self.estimator.coef_, (self.order, n_dof)).T
                else:
                    coefs_ = np.zeros((self.order, n_dof))
                    coefs_[:, np.arange(n_dof) != i] = np.reshape(self.estimator.coef_, 
                                                                  (self.order, n_dof - 1))
                    coefs_ = coefs_.T

                coefs.append(np.fliplr(coefs_))
                if hasattr(self.estimator, 'intercept_'):
                    intercept[idx] = self.estimator.intercept_
                if hasattr(self.estimator, 'scores_'):
                    # Match scores up with supports
                    scores.append({'scores':self.estimator.scores_, 'supports':self.estimator.supports_})

            coefs = np.array(coefs)

            # Gather coefficients onto the root subcomm
            # This check is a bit of a hack (?)
            if self.root_group.rank >= 0:
                self.coef_ = Gatherv_rows(coefs, self.rootcomm, root=0)
                self.scores_ = self.rootcomm.gather(scores)
            if self.comm.rank == 0:
                # Re-order coefficients so model order comes first
                self.coef_ = np.transpose(self.coef_, axes=(2, 0, 1))

        else:

            # Statistics to track
            self.intercept_ = np.zeros(n_dof)
            self.coef_ = np.zeros((n_dof, n_dof, self.order))
            self.scores_ = []
            self.supports_ = []

            for i in range(n_dof):
                print('Row %d' % i)
                xx, yy = _form_var_problem(y, i, self.order, self.self_regress)
                self.estimator.fit(xx, yy)

                if self.self_regress:
                    coefs = np.reshape(self.estimator.coef_, (self.order, n_dof)).T
                else:
                    coefs = np.zeros((self.order, n_dof))
                    coefs[:, np.arange(n_dof) != i] = np.reshape(self.estimator.coef_, 
                                                                 (self.order, n_dof - 1))
                    coefs = coefs.T

                self.coef_[i, ...] = np.fliplr(coefs)

                if hasattr(self.estimator, 'intercept_'):
                    self.intercept_[i] = self.estimator.intercept_
                if hasattr(self.estimator, 'scores_'):
                    self.scores_.append(self.estimator.scores_) 
                if hasattr(self.estimator, 'supports_'):
                    self.supports_.append(self.estimator.supports_)

            # Re-order coefficients so model order comes first
            self.coef_ = np.transpose(self.coef_, axes=(2, 0, 1))
        # # Re-order coefficients so AR(1) coefficient comes first in
        # # the last axis
        # self.coef_ = np.transpose(self.coef_, axes=(1, 0, 2))
        # self.coef_ = np.flip(self.coef_, axis=-1)

    # Forecast the time evolution of y
    def predict(self, y):

        y_pred = np.zeros((y.shape[0] - self.order, y.shape[1]))

        for i in range(y_pred.shape[0]):
            y_pred[i, :] = np.sum(np.vstack([y[i + self.order - j - 1] @ self.coef_[j, ...] 
                                  for j in range(self.coef_.shape[0])]), axis=0)
        # # Return a trimmed version of y for proper comparison with y_pred
        return y_pred, y[self.order:]

    def score(self, y, metric='r2'):
        y_pred, y = self.predict(y) 
        if metric == 'r2':
            score = r2_score(y, y_pred)

        return score


class UoIVAR_Estimator(UoI_NCV):

    def __init__(self, penalty, random_state, fit_intercept=False,
                 fit_type='union_only', resample_type='fixed_order', L=None,
                 **uoi_kwargs):
        
        self.fit_type = fit_type
        self.resample_type = resample_type
        self.L = L
        self.fit_type = fit_type

        if fit_type == 'union_only':

            uoi_kwargs['n_boots_sel'] = 1
            uoi_kwargs['stability_selection'] = 1.
            uoi_kwargs['selection_frac'] = 1.

        super(UoIVAR_Estimator, self).__init__(penalty=penalty,
                                               random_state=random_state,
                                               **uoi_kwargs)

        self._estimation_lm = VAR_OLS_Wrapper(fit_intercept=fit_intercept)

    def _resample(self, idxs, *args, **kwargs):
        """Modify default resampling behavior"""
        if self.resample_type == 'fixed_order':
            return super(UoIVAR_Estimator, self)._resample(idxs, *args, **kwargs)

        elif self.resample_type == 'block':
            return resample('block', idxs, replace=True, 
                            random_state=self.random_state,
                                   L=self.L)

    def _fit_intercept(self, X, y):

        n_dof = X.shape[1]

        if self.fit_intercept:
            self.intercept_ = (y.mean(axis=0) -
                               np.dot(X.mean(axis=0), self.coef_.T))
        else:
            self.intercept_ = np.zeros(n_dof)

    def fit(self, X, y, savepath=None):
        super(UoIVAR_Estimator, self).fit(X, y)
        
        if savepath is not None:

            if self.comm is not None:
                if self.comm.rank == 0:
                    print('Saving!')
                    with open(savepath, 'wb') as f:
                        f.write(pickle.dumps(self.coef_))
                        f.write(pickle.dumps({'supports':self.supports_}))
            else:
                with open(savepath, 'wb') as f:
                    f.write(pickle.dumps(self.coef_))
                    f.write(pickle.dumps({'scores':self.scores_, 'supports':self.supports_}))


class NCV_VAR_Estimator(PycWrapper):

    def __init__(self, random_state=None, alphas=None, fit_intercept=False, 
                 max_iter=1000, penalty='l1', selection_method='BIC'): 

        self.random_state = random_state

        if selection_method == 'CV':
            self.cross_validate = True
        else:
            self.cross_validate = False

        self.selection_method = selection_method

        super(NCV_VAR_Estimator, self).__init__(alphas=alphas, fit_intercept=fit_intercept, 
                                                max_iter=max_iter, penalty=penalty)
    def fit(self, X, y):
        # Standardize by default
        X_scaler = StandardScaler(with_mean=self.fit_intercept)
        X = X_scaler.fit_transform(X)
        y_scaler = StandardScaler(with_mean=self.fit_intercept)
        y = y_scaler.fit_transform(y.reshape(-1, 1))

        alphas = _alpha_grid(X, y)
        self.set_params(alphas=alphas)

        super(NCV_VAR_Estimator, self).fit(X, y, cross_validate=self.cross_validate)
        self.all_coef = self.coef_
        self.select(X, y)

    def select(self, X, y):

        if self.selection_method == 'BIC':

            n_models = self.coef_.shape[0]
            scores = np.zeros(n_models)
            y_pred = self.predict(X)
            for model_idx in range(n_models):
                ll = log_likelihood_glm('normal', y, y_pred[:, model_idx])
                scores[model_idx] = BIC(ll, X.shape[0], X.shape[1]) 
            self.scores_ = scores
            self.coef_ = self.coef_[np.argmin(scores), :]
            self.intercept_ = self.intercept_[np.argmin(scores)]

        elif self.selection_method == 'CV':

            n_folds = self.coef_.shape[0]
            n_models = self.coef_.shape[1]
            scores = np.zeros((n_folds, n_models))

            for fold in range(n_folds):
                Xtest = X[self.test_folds[fold]]
                ytest = y[self.test_folds[fold]]
                y_pred = self.predict(X, fold=fold)
                scores[fold, :] = [r2_score(y, y_pred[:, i]) for i in range(y_pred.shape[1])]

            # Average over folds
            scores = np.mean(scores, axis=0)
            self.scores_ = scores

            # refit with the best model
            best_model = np.argmax(scores)

            alphas = self.alphas
            self.alphas = np.array([alphas[best_model]])
            super(NCV_VAR_Estimator, self).fit(X, y, cross_validate=False)
            self.alphas = alphas

class VAR_OLS_Wrapper(LinearRegression):

    def __init__(self, fit_intercept=False, normalize=False, 
                 copy_X=True, n_jobs=None, standalone=False):

        self.standalone = standalone
        super(VAR_OLS_Wrapper, self).__init__(fit_intercept=fit_intercept,
                                              normalize=normalize,  
                                              copy_X=copy_X,
                                              n_jobs=n_jobs)

    def fit(self, X, y, coef_mask=None):
        n_features = X.shape[1]
        if coef_mask is not None:
            X = X[:, coef_mask]

        super(VAR_OLS_Wrapper, self).fit(X, y)

        if self.standalone:
            coef = np.zeros(n_features)
            coef[coef_mask] = self.coef_
            self.coef_ = coef

    def predict(self, X, y, coef_mask=None):

        if coef_mask is not None:
            X = X[:, coef_mask]

        y_pred = X @ self.coef_

        return y, y_pred

def form_lag_matrix(X, T, y=None, stride=1, stride_tricks=True,
                    writeable=False):
    
    # Do nothing
    if T == 0:
        if y is not None: 
            return X, y
        else:
            return X

    if X.ndim == 2:
        return _form_lag_matrix(X, T, y=y, stride=stride, stride_tricks=stride_tricks,
                                writeable=writeable)
    elif X.ndim == 3:
        # if y is not None and y.ndim !=2:
        #     raise ValueError('y passed in with non-standard dimension.')
        # Separately lag each trial and then concatenate
        xx = []
        yy = []

        for i in range(X.shape[0]):
            # print('Forming lag matrix')
            if y is not None:
                xxlag, yylag = _form_lag_matrix(X[i, ...], T, y=y[i, ...], stride=stride, 
                                                stride_tricks=stride_tricks,
                                                writeable=writeable) 
            else:
                xxlag, yylag = _form_lag_matrix(X[i, ...], T, None, stride=stride, 
                                                stride_tricks=stride_tricks,
                                                writeable=writeable) 

            xx.append(xxlag)
            yy.append(yylag)

        return xx, yy

def _form_lag_matrix(X, T, y=None, stride=1, stride_tricks=True, 
                    writeable=False):
    """Form the data matrix with `T` lags.

    Parameters
    ----------
    X : ndarray (n_time, N)
        Timeseries with no lags.
    T : int
        Number of lags.
    stride : int
        Number of original samples to move between lagged samples.
    stride_tricks : bool
        Whether to use numpy stride tricks to form the lagged matrix or create
        a new array. Using numpy stride tricks can can lower memory usage, especially for
        large `T`. If `False`, a new array is created.
    writeable : bool
        For testing. You should not need to set this to True. This function uses stride tricks
        to form the lag matrix which means writing to the array will have confusing behavior.
        If `stride_tricks` is `False`, this flag does nothing.

    Returns
    -------
    X_with_lags : ndarray (n_lagged_time, N * T)
        Timeseries with lags.
    """

    if not isinstance(stride, int) or stride < 1:
        raise ValueError('stride should be an int and greater than or equal to 1.')
    N = X.shape[1]
    n_lagged_samples = (len(X) - T) // stride + 1
    if n_lagged_samples < 1:
        raise ValueError('T is too long for a timeseries of length {}.'.format(len(X)))
    if stride_tricks:
        X = np.asarray(X, dtype=float, order='C')
        shape = (n_lagged_samples, N * T)
        strides = (X.strides[0] * stride,) + (X.strides[-1],)
        X_with_lags = as_strided(X, shape=shape, strides=strides, writeable=writeable)
    else:
        X_with_lags = np.zeros((n_lagged_samples, N * T))
        for i in range(n_lagged_samples):
            X_with_lags[i, :] = X[i * stride:i * stride + T, :].flatten()

    # Trim off the last index
    X_with_lags = X_with_lags[:-1, :]

    # Trim off the beginning
    if y is not None:
        y = y[y.shape[0] - n_lagged_samples + 1:]

    return X_with_lags, y

# Given a time series, return a sequence of regression problems 
def _form_var_problem(y, i, T, self_regress=False):

    if self_regress:
        xx, yy = form_lag_matrix(y, T, y[..., i])
    else:
        xx, yy = form_lag_matrix(y[..., np.arange(y.shape[-1]) != i], T, y[..., i])                

    xx = np.array(xx)
    yy = np.array(yy)

    xx = np.reshape(xx, (-1, xx.shape[-1]))
    yy = np.reshape(yy, (-1,))[:, np.newaxis]

    return xx, yy
