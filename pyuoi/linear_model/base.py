import abc as _abc
import six as _six
import numpy as np
import pdb

from sklearn.linear_model.base import (
    LinearModel, _preprocess_data, SparseCoefMixin)
from sklearn.metrics import r2_score, accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.utils import check_X_y
from sklearn.feature_selection import mutual_info_regression

from pyuoi import utils
from pyuoi.mpi_utils import (Gatherv_rows, Bcast_from_root)

from .utils import stability_selection_to_threshold, intersection


class AbstractUoILinearModel(
        _six.with_metaclass(_abc.ABCMeta, LinearModel, SparseCoefMixin)):
    """An abstract base class for UoI linear model classes

    See Bouchard et al., NIPS, 2017, for more details on the Union of
    Intersections framework.

    Parameters
    ----------
    n_boots_sel : int, default 48
        The number of data bootstraps to use in the selection module.
        Increasing this number will make selection more strict.

    n_boots_est : int, default 48
        The number of data bootstraps to use in the estimation module.
        Increasing this number will relax selection and decrease variance.

    selection_frac : float, default 0.9
        The fraction of the dataset to use for training in each resampled
        bootstrap, during the selection module. Small values of this parameter
        imply larger "perturbations" to the dataset.

    estimation_frac : float, default 0.9
        The fraction of the dataset to use for training in each resampled
        bootstrap, during the estimation module. The remaining data is used
        to obtain validation scores. Small values of this parameters imply
        larger "perturbations" to the dataset. IGNORED - Leaving this here
        to double check later

    stability_selection : int, float, or array-like, default 1
        If int, treated as the number of bootstraps that a feature must
        appear in to guarantee placement in selection profile. If float,
        must be between 0 and 1, and is instead the proportion of
        bootstraps. If array-like, must consist of either ints or floats
        between 0 and 1. In this case, each entry in the array-like object
        will act as a separate threshold for placement in the selection
        profile.

    copy_X : boolean, default True
        If True, X will be copied; else, it may be overwritten.

    fit_intercept : boolean, default True
        Whether to calculate the intercept for this model. If set
        to False, no intercept will be used in calculations
        (e.g. data is expected to be already centered).

    normalize : boolean, default False
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True, the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.

    random_state : int, RandomState instance or None, default None
        The seed of the pseudo random number generator that selects a random
        feature to update.  If int, random_state is the seed used by the random
        number generator; If RandomState instance, random_state is the random
        number generator; If None, the random number generator is the
        RandomState instance used by `np.random`.

    Attributes
    ----------
    coef_ : array, shape (n_features,) or (n_targets, n_features)
        Estimated coefficients for the linear regression problem.

    intercept_ : float
        Independent term in the linear model.

    supports_ : array, shape
        boolean array indicating whether a given regressor (column) is selected
        for estimation for a given regularization parameter value (row).
    """

    def __init__(self, n_boots_sel=48, n_boots_est=48, selection_frac=0.9,
                 estimation_frac=0.9, stability_selection=1.,
                 random_state=None, comm=None):
        # data split fractions
        self.selection_frac = selection_frac
        self.estimation_frac = estimation_frac
        # number of bootstraps
        self.n_boots_sel = n_boots_sel
        self.n_boots_est = n_boots_est
        # other hyperparameters
        self.stability_selection = stability_selection
        # preprocessing
        self.comm = comm
        if isinstance(random_state, int):
            # make sure ranks use different seed
            if self.comm is not None:
                random_state += self.comm.rank
            self.random_state = np.random.RandomState(random_state)
        else:
            if random_state is None:
                self.random_state = np.random
            else:
                self.random_state = random_state

        # extract selection thresholds from user provided stability selection
        self.selection_thresholds_ = stability_selection_to_threshold(
            self.stability_selection, self.n_boots_sel)

        self.n_supports_ = None

    @_abc.abstractproperty
    def selection_lm(self):
        pass

    @_abc.abstractproperty
    def estimation_lm(self):
        pass

    @_abc.abstractproperty
    def estimation_score(self):
        pass

    @_abc.abstractmethod
    def get_reg_params(self):
        pass

    @_abc.abstractstaticmethod
    def score_predictions(metric, y_true, y_pred, supports):
        pass

    @_abc.abstractmethod
    def intersect(self, coef, thresholds):
        """Intersect coefficients accross all thresholds"""
        pass

    @_abc.abstractmethod
    def preprocess_data(self, X, y):
        """

        """
        pass

    @_abc.abstractmethod
    def get_n_coef(self, X, y):
        """"Return the number of coefficients that will be estimated

        This should return the total number of coefficients estimated,
        accounting for all coefficients for multi-target regression or
        multi-class classification.
        """
        pass

    @_abc.abstractmethod
    def _fit_intercept(self, y):
        """"Fit a model with an intercept and fixed coefficients.

        This is used to re-fit the intercept after the coefficients are
        estimated.
        """
        pass

    @_abc.abstractmethod
    def _fit_intercept_no_features(self, y):
        """"Fit a model with only an intercept.

        This is used in cases where the model has no support selected.
        """
        pass

    def fit(self, X, y, stratify=None, forward_selection = False, verbose=False):
        """Fit data according to the UoI algorithm.

        Parameters
        ----------
        X : ndarray or scipy.sparse matrix, (n_samples, n_features)
            The design matrix.

        y : ndarray, shape (n_samples,)
            Response vector. Will be cast to X's dtype if necessary.
            Currently, this implementation does not handle multiple response
            variables.

        stratify : array-like or None, default None
            Ensures groups of samples are alloted to training/test sets
            proportionally. Labels for each group must be an int greater
            than zero. Must be of size equal to the number of samples, with
            further restrictions on the number of groups.

        verbose : boolean
            A switch indicating whether the fitting should print out messages
            displaying progress. Utilizes tqdm to indicate progress on
            bootstraps.
        """

        # extract model dimensions
        n_samples, n_coef = self.get_n_coef(X, y)
        n_features = X.shape[1]

        ####################
        # Selection Module #
        ####################
        # choose the regularization parameters for selection sweep
        self.reg_params_ = self.get_reg_params(X, y)
        self.n_reg_params_ = len(self.reg_params_)

        rank = 0
        size = 1
        if self.comm is not None:
            rank = self.comm.rank
            size = self.comm.size

        # initialize selection
        if size > self.n_boots_sel:
            tasks = np.array_split(np.arange(self.n_boots_sel *
                                             self.n_reg_params_), size)[rank]
            selection_coefs = np.empty((tasks.size, n_coef))
            my_boots = dict((task_idx // self.n_reg_params_, None)
                            for task_idx in tasks)
        else:
            # split up bootstraps into processes
            tasks = np.array_split(np.arange(self.n_boots_sel),
                                   size)[rank]
            selection_coefs = np.empty((tasks.size, self.n_reg_params_,
                                        n_coef))
            my_boots = dict((task_idx, None) for task_idx in tasks)

        for boot in range(self.n_boots_sel):
            if self.comm is not None:
                if rank == 0:
                    rvals = train_test_split(np.arange(X.shape[0]),
                                             test_size=1 - self.selection_frac,
                                             stratify=stratify,
                                             random_state=self.random_state)
                else:
                    rvals = [None] * 2
                rvals = [Bcast_from_root(rval, self.comm, root=0)
                         for rval in rvals]
                if boot in my_boots.keys():
                    my_boots[boot] = rvals
            else:
                my_boots[boot] = train_test_split(
                    np.arange(X.shape[0]),
                    test_size=1 - self.selection_frac,
                    stratify=stratify,
                    random_state=self.random_state)

        # iterate over bootstraps
        curr_boot_idx = None
        for ii, task_idx in enumerate(tasks):
            if size > self.n_boots_sel:
                boot_idx = task_idx // self.n_reg_params_
                reg_idx = task_idx % self.n_reg_params_
                my_reg_params = [self.reg_params_[reg_idx]]
            else:
                boot_idx = task_idx
                my_reg_params = self.reg_params_
            # Never warm start across bootstraps
            if (curr_boot_idx != boot_idx) and hasattr(self.selection_lm,
                                                       'coef_'):
                self.selection_lm.coef_[:] = 0.
            curr_boot_idx = boot_idx

            # draw a resampled bootstrap
            idxs_train, idxs_test = my_boots[boot_idx]
            X_rep = X[idxs_train]
            X_test = X[idxs_test]
            y_rep = y[idxs_train]
            y_test = y[idxs_test]

            # fit the coefficients
            selection_coefs[ii] = np.squeeze(
                self.uoi_selection_sweep(X_rep, y_rep, my_reg_params))

        # if distributed, gather selection coefficients to 0,
        # perform intersection, and broadcast results
        if self.comm is not None:
            selection_coefs = Gatherv_rows(selection_coefs, self.comm, root=0)
            if rank == 0:
                if size > self.n_boots_sel:
                    selection_coefs = selection_coefs.reshape(
                        self.n_boots_sel,
                        self.n_reg_params_,
                        n_coef)
                supports = self.intersect(
                    selection_coefs,
                    self.selection_thresholds_).astype(int)
            else:
                supports = None
            supports = Bcast_from_root(supports, self.comm, root=0)
            self.supports_ = supports.astype(bool)
        else:
            self.supports_ = self.intersect(selection_coefs,
                                            self.selection_thresholds_)

        self.n_supports_ = self.supports_.shape[0]

        #####################
        # Estimation Module #
        #####################
        # set up data arrays
        tasks = np.array_split(np.arange(self.n_boots_est *
                                         self.n_supports_), size)[rank]
        my_boots = dict((task_idx // self.n_supports_, None)
                        for task_idx in tasks)
        estimates = np.zeros((tasks.size, n_coef))

        for boot in range(self.n_boots_est):
            if self.comm is not None:
                if rank == 0:
                    rvals = train_test_split(np.arange(X.shape[0]),
                                             test_size=1 - self.selection_frac,
                                             stratify=stratify,
                                             random_state=self.random_state)
                else:
                    rvals = [None] * 2
                rvals = [Bcast_from_root(rval, self.comm, root=0)
                         for rval in rvals]
                if boot in my_boots.keys():
                    my_boots[boot] = rvals
            else:
                my_boots[boot] = train_test_split(
                    np.arange(X.shape[0]),
                    test_size=1 - self.selection_frac,
                    stratify=stratify,
                    random_state=self.random_state)

        # score (r2/AIC/AICc/BIC) for each bootstrap for each support
            scores = np.zeros(tasks.size)

        n_tile = n_coef // n_features
        # iterate over bootstrap samples and supports
        for ii, task_idx in enumerate(tasks):
            boot_idx = task_idx // self.n_supports_
            support_idx = task_idx % self.n_supports_
            support = self.supports_[support_idx]
            # draw a resampled bootstrap
            idxs_train, idxs_test = my_boots[boot_idx]
            X_rep = X[idxs_train]
            X_test = X[idxs_test]
            y_rep = y[idxs_train]
            y_test = y[idxs_test]
            if np.any(support):

                # compute ols estimate
                self.estimation_lm.fit(X_rep[:, support], y_rep)

                estimates[ii, np.tile(support, n_tile)] = \
                    self.estimation_lm.coef_.ravel()

                scores[ii] = self.score_predictions(
                    metric=self.estimation_score,
                    fitter=self.estimation_lm,
                    X=X_test, y=y_test,
                    support=support)
            else:
                fitter = self._fit_intercept_no_features(y_rep)
                scores[ii] = self.score_predictions(
                    metric=self.estimation_score,
                    fitter=fitter,
                    X=np.zeros_like(X_test), y=y_test,
                    support=np.zeros(X_test.shape[1], dtype=bool))

        if self.comm is not None:
            scores = Gatherv_rows(send=scores, comm=self.comm,
                                      root=0)
            # Distribute forward selection loops across processes
            if forward_selection:
                # Re-distribute
                scores = Bcast_from_root(scores, comm=self.comm, root = 0)
                
                scores = scores.reshape(self.n_boots_est, self.n_supports_)
                self.rp_max_idx_ = np.argmax(scores, axis=1)

                # Chunk
                numproc = self.comm.Get_size()
                chunk_param_list = np.array_split(np.arange(self.rp_max_idx_.size), numproc)
                chunk_idx = rank

                # Assemble a separate set of estimates
                forward_estimates = np.zeros((len(chunk_param_list[chunk_idx]), n_coef))

                for i, max_idx in enumerate(self.rp_max_idx_[chunk_param_list[chunk_idx]]):

                    idx = chunk_param_list[chunk_idx][i]

                    support_ = np.copy(self.supports_[max_idx])
                    current_score = scores[idx, max_idx]

                    # Generate new bootstrap sample
                    idxs_train, idxs_test = train_test_split(
                    np.arange(X.shape[0]),
                    test_size=1 - self.selection_frac,
                    stratify=stratify,
                    random_state=self.random_state)

                    X_rep = X[idxs_train]
                    X_test = X[idxs_test]
                    y_rep = y[idxs_train]
                    y_test = y[idxs_test]

                    while True:

                            present_features = np.count_nonzero(1 * support_)
                            print('%d/%d features present' % (present_features, n_features))

                            if present_features == n_features:
                                break

                            # Add the feature that has the most 
                            # mutual information with the response
                            mutual_infos = mutual_info_regression(
                                           X_rep[:, np.invert(support_)], y_rep)
                            support_[np.argmax(mutual_infos)] = True

                            # Calculate the score of the model with the new
                            # feature added
                            self.estimation_lm.fit(X_rep[:, support_], y_rep)
                            new_score = self.score_predictions(
                            metric = self.estimation_score,
                            fitter = self.estimation_lm,
                            X = X_test,
                            y = y_test,
                            support = support_)

                            print('New_score: %f' % new_score)

                            if new_score <= current_score:
                                support_[np.argmax(mutual_infos)] = False
                                break

                            current_score = new_score

                    # Refit with the final support set
                    self.estimation_lm.fit(X_rep[:, support_], y_rep)
                    forward_estimates[i, support_] = self.estimation_lm.coef_

                forward_estimates = Gatherv_rows(send=forward_estimates, comm=self.comm, root=0)
                coef = None
                if rank == 0:

                        forward_estimates.reshape(self.n_boots_est, n_coef)
                        coef = np.median(forward_estimates,
                                                axis = 0).reshape(n_tile, n_coef)

                self.coef_ = Bcast_from_root(coef, self.comm, root = 0)

            else: # No forward selection
                estimates = Gatherv_rows(send=estimates, comm=self.comm,
                                             root=0)
                if rank == 0:

                        estimates = estimates.reshape(self.n_boots_est,
                                              self.n_supports_, n_coef)
                        scores = scores.reshape(self.n_boots_est, self.n_supports_)
                        self.rp_max_idx_ = np.argmax(scores, axis=1)

                        # extract the estimates over bootstraps from model with best
                        # regularization parameter value
                        best_estimates = self.estimates_[np.arange(self.n_boots_est),
                                                         self.rp_max_idx_]

                        # take the median across estimates for the final, bagged estimate
                        coef = np.median(best_estimates,
                                     axis=0).reshape(n_tile, n_coef)

                self.estimates_ = Bcast_from_root(estimates, self.comm, root=0)
                self.scores_ = Bcast_from_root(scores, self.comm, root=0)
                self.coef_ = Bcast_from_root(coef, self.comm, root=0)
                self.rp_max_idx_ = self.comm.bcast(self.rp_max_idx_, root=0)
        else: # No MPI
            self.estimates_ = estimates.reshape(self.n_boots_est,
                                                self.n_supports_, n_coef)
            self.scores_ = scores.reshape(self.n_boots_est, self.n_supports_)
            self.rp_max_idx_ = np.argmax(self.scores_, axis=1)
            # Iteratively expand these best performing support sets
            if forward_selection:
                # Assemble a separate set of estimates
                forward_estimates = np.zeros((self.n_boots_est, n_coef))

                for i, max_idx in enumerate(self.rp_max_idx_):

                    support_ = np.copy(self.supports_[max_idx])
                    current_score = self.scores_[i, max_idx]

                    # Generate new bootstrap sample
                    idxs_train, idxs_test = train_test_split(
                    np.arange(X.shape[0]),
                    test_size=1 - self.selection_frac,
                    stratify=stratify,
                    random_state=self.random_state)

                    X_rep = X[idxs_train]
                    X_test = X[idxs_test]
                    y_rep = y[idxs_train]
                    y_test = y[idxs_test]

                    while True:

                        present_features = np.count_nonzero(1 * support_)
                        print('%d/%d features present' % (present_features, n_features))

                        if present_features == n_features:
                            break

                        # Add the feature that has the most 
                        # mutual information with the response
                        mutual_infos = mutual_info_regression(
                                       X_rep[:, np.invert(support_)], y_rep)
                        support_[np.argmax(mutual_infos)] = True

                        # Calculate the score of the model with the new
                        # feature added
                        self.estimation_lm.fit(X_rep[:, support_], y_rep)
                        new_score = self.score_predictions(
                        metric = self.estimation_score,
                        fitter = self.estimation_lm,
                        X = X_test,
                        y = y_test,
                        support = support_)

                        print('New_score: %f' % new_score)

                        if new_score <= current_score:
                            support_[np.argmax(mutual_infos)] = False
                            break

                        current_score = new_score

                    # Refit with the final support set
                    self.estimation_lm.fit(X_rep[:, support_], y_rep)
                    forward_estimates[i, support_] = self.estimation_lm.coef_

                self.coef_ = np.median(forward_estimates,
                                        axis = 0).reshape(n_tile, n_coef)
            else: # No MPI, no forward selection
                # extract the estimates over bootstraps from model with best
                # regularization parameter value
                best_estimates = self.estimates_[np.arange(self.n_boots_est),
                                                 self.rp_max_idx_, :]

                # take the median across estimates for the final, bagged estimate
                self.coef_ = np.median(best_estimates,
                                       axis=0).reshape(n_tile, n_coef)

        return self

    def uoi_selection_sweep(self, X, y, reg_param_values):
        """Perform selection regression on a dataset over a sweep of
        regularization parameter values.

        Parameters
        ----------
        X : ndarray or scipy.sparse matrix, (n_samples, n_features)
            The design matrix.

        y : ndarray, shape (n_samples,)
            Response vector.

        reg_param_values: list of dicts
            A list of dictionaries containing the regularization parameter
            values to iterate over.

        Returns
        -------
        coefs : nd.array, shape (n_param_values, n_features)
            Predicted parameter values for each regularization strength.
        """

        n_param_values = len(reg_param_values)
        n_samples, n_coef = self.get_n_coef(X, y)

        coefs = np.zeros((n_param_values, n_coef))

        # apply the selection regression to bootstrapped datasets
        for reg_param_idx, reg_params in enumerate(reg_param_values):
            # reset the regularization parameter
            self.selection_lm.set_params(**reg_params)
            # rerun fit
            self.selection_lm.fit(X, y)
            # store coefficients
            coefs[reg_param_idx] = self.selection_lm.coef_.ravel()

        return coefs


class AbstractUoILinearRegressor(
        _six.with_metaclass(_abc.ABCMeta, AbstractUoILinearModel)):
    """An abstract base class for UoI linear regression classes.

    See Bouchard et al., NIPS, 2017, for more details on the Union of
    Intersections framework.
    """

    __valid_estimation_metrics = ('r2', 'AIC', 'AICc', 'BIC')

    def __init__(self, n_boots_sel=48, n_boots_est=48, selection_frac=0.9,
                 estimation_frac=0.9, stability_selection=1.,
                 estimation_score='r2', copy_X=True, fit_intercept=True,
                 normalize=True, random_state=None, max_iter=1000,
                 comm=None):
        super(AbstractUoILinearRegressor, self).__init__(
            n_boots_sel=n_boots_sel,
            n_boots_est=n_boots_est,
            selection_frac=selection_frac,
            estimation_frac=estimation_frac,
            stability_selection=stability_selection,
            random_state=random_state,
            comm=comm,
        )
        self.copy_X = copy_X
        self.fit_intercept = fit_intercept
        self.normalize = normalize

        if estimation_score not in self.__valid_estimation_metrics:
            raise ValueError(
                "invalid estimation metric: '%s'" % estimation_score)

        self.__estimation_score = estimation_score

    def preprocess_data(self, X, y):
        return _preprocess_data(
            X, y, fit_intercept=self.fit_intercept, normalize=self.normalize,
            copy=self.copy_X
        )

    def get_n_coef(self, X, y):
        """"Return the number of coefficients that will be estimated

        This should return the shape of X.
        """
        return X.shape

    def intersect(self, coef, thresholds):
        """Intersect coefficients accross all thresholds"""
        return intersection(coef, thresholds)

    @property
    def estimation_score(self):
        return self.__estimation_score

    @staticmethod
    def score_predictions(metric, fitter, X, y, support):
        """Score, according to some metric, predictions provided by a model.

        the resulting score will be negated if an information criterion is
        specified

        Parameters
        ----------
        metric : string
            The type of score to run on the prediction. Valid options include
            'r2' (explained variance), 'BIC' (Bayesian information criterion),
            'AIC' (Akaike information criterion), and 'AICc' (corrected AIC).

        y_true : array-like
            The true response variables.

        y_pred : array-like
            The predicted response variables.

        supports: array-like
            The value of the supports for the model that was used to generate
            *y_pred*.

        Returns
        -------
        score : float
            The score.
        """
        y_pred = fitter.predict(X[:, support])
        if metric == 'r2':
            score = r2_score(y, y_pred)
        else:
            ll = utils.log_likelihood_glm(model='normal',
                                          y_true=y,
                                          y_pred=y_pred)
            n_features = np.count_nonzero(support)
            n_samples = y.size
            if metric == 'BIC':
                score = utils.BIC(ll, n_features, n_samples)
            elif metric == 'AIC':
                score = utils.AIC(ll, n_features)
            elif metric == 'AICc':
                score = utils.AICc(ll, n_features, n_samples)
            else:
                raise ValueError(metric + ' is not a valid option.')
            # negate the score since lower information criterion is preferable
            score = -score
        return score

    def fit(self, X, y, forward_selection = False, stratify=None, verbose=False):
        """Fit data according to the UoI algorithm.

        Additionaly, perform X-y checks, data preprocessing, and setting
        intercept.

        Parameters
        ----------
        X : ndarray or scipy.sparse matrix, (n_samples, n_features)
            The design matrix.

        y : ndarray, shape (n_samples,)
            Response vector. Will be cast to X's dtype if necessary.
            Currently, this implementation does not handle multiple response
            variables.

        stratify : array-like or None, default None
            Ensures groups of samples are alloted to training/test sets
            proportionally. Labels for each group must be an int greater
            than zero. Must be of size equal to the number of samples, with
            further restrictions on the number of groups.

        forward_selection : Attempt to expand the suppport set in the 
        estimation module

        verbose : boolean
            A switch indicating whether the fitting should print out messages
            displaying progress. Utilizes tqdm to indicate progress on
            bootstraps.
        """
        # perform checks
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'],
                         y_numeric=True, multi_output=True)
        # preprocess data
        X, y, X_offset, y_offset, X_scale = self.preprocess_data(X, y)
        super(AbstractUoILinearRegressor, self).fit(X, y, stratify=stratify,
                                                forward_selection = forward_selection,
                                                verbose=verbose)

        self._fit_intercept(X_offset, y_offset, X_scale)
        self.coef_ = np.squeeze(self.coef_)
        return self

    def _fit_intercept_no_features(self, y):
        """"Fit a model with only an intercept.

        This is used in cases where the model has no support selected.
        """
        return LinearInterceptFitterNoFeatures(y)


class LinearInterceptFitterNoFeatures(object):
    def __init__(self, y):
        self.intercept_ = y.mean()

    def predict(self, X):
        n_samples = X.shape[0]
        return np.tile(self.intercept_, n_samples)


class AbstractUoILinearClassifier(
        _six.with_metaclass(_abc.ABCMeta, AbstractUoILinearModel)):
    """An abstract base class for UoI linear classifier classes.

    See Bouchard et al., NIPS, 2017, for more details on the Union of
    Intersections framework.
    """

    __valid_estimation_metrics = ('acc', 'log')

    def __init__(self, n_boots_sel=48, n_boots_est=48, selection_frac=0.9,
                 estimation_frac=0.9, stability_selection=1.,
                 estimation_score='acc', multi_class='ovr',
                 copy_X=True, fit_intercept=True, normalize=True,
                 random_state=None, max_iter=1000,
                 comm=None):
        super(AbstractUoILinearClassifier, self).__init__(
            n_boots_sel=n_boots_sel,
            n_boots_est=n_boots_est,
            selection_frac=selection_frac,
            estimation_frac=estimation_frac,
            stability_selection=stability_selection,
            random_state=random_state,
            comm=comm,
        )
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X

        if estimation_score not in self.__valid_estimation_metrics:
            raise ValueError(
                "invalid estimation metric: '%s'" % estimation_score)
        self.__estimation_score = estimation_score

    def get_n_coef(self, X, y):
        """"Return the number of coefficients that will be estimated

        This should return the shape of X if doing binary classification,
        else return (X.shape[0], X.shape[1]*n_classes).
        """
        n_samples, n_coef = X.shape
        self._n_classes = len(np.unique(y))
        self._labels = np.unique(y)
        if self._n_classes > 2:
            n_coef = n_coef * self._n_classes
        return n_samples, n_coef

    def intersect(self, coef, thresholds):
        """Intersect coefficients accross all thresholds

        This implementation will account for multi-class classification.
        """
        supports = intersection(coef, thresholds)
        ret = supports
        if self._n_classes > 2:
            # for each support, figure out which variables
            # are used
            ret = list()
            n_coef = supports[0].shape[0] // self._n_classes
            shape = (self._n_classes, n_coef)
            for supp in supports:
                ret.append(np.logical_or(*supp.reshape(shape)))
            uniq = set()
            for sup in ret:
                uniq.add(tuple(sup))
            ret = np.array(list(uniq))
        return ret

    @staticmethod
    def preprocess_data(self, X, y):
        return _preprocess_data(
            X, y, fit_intercept=self.fit_intercept, normalize=self.normalize,
            copy=self.copy_X
        )

    @property
    def estimation_score(self):
        return self.__estimation_score

    @staticmethod
    def score_predictions(metric, fitter, X, y, support):
        """Score, according to some metric, predictions provided by a model.

        the resulting score will be negated if an information criterion is
        specified

        Parameters
        ----------
        metric : string
            The type of score to run on the prediction. Valid options include
            'r2' (explained variance), 'BIC' (Bayesian information criterion),
            'AIC' (Akaike information criterion), and 'AICc' (corrected AIC).

        y_true : array-like
            The true response variables.

        y_pred : array-like
            The predicted response variables.

        supports: array-like
            The value of the supports for the model that was used to generate
            *y_pred*.

        Returns
        -------
        score : float
            The score.
        """
        if metric == 'acc':
            y_pred = fitter.predict(X[:, support])
            score = accuracy_score(y, y_pred)
        else:
            y_pred = fitter.predict_proba(X[:, support])
            ll = -log_loss(y, y_pred)
            if metric == 'log':
                score = ll
            else:
                n_features = np.count_nonzero(support)
                n_samples = y.size
                if metric == 'BIC':
                    score = utils.BIC(ll, n_features, n_samples)
                elif metric == 'AIC':
                    score = utils.AIC(ll, n_features)
                elif metric == 'AICc':
                    score = utils.AICc(ll, n_features, n_samples)
                else:
                    raise ValueError(metric + ' is not a valid metric.')
                # negate the score since lower information criterion is
                # preferable
                score = -score

        return score

    def fit(self, X, y, stratify=None, verbose=False):
        """Fit data according to the UoI algorithm.

        Additionaly, perform X-y checks, data preprocessing, and setting
        intercept.

        Parameters
        ----------
        X : ndarray or scipy.sparse matrix, (n_samples, n_features)
            The design matrix.

        y : ndarray, shape (n_samples,)
            Response vector. Will be cast to X's dtype if necessary.
            Currently, this implementation does not handle multiple response
            variables.

        stratify : array-like or None, default None
            Ensures groups of samples are alloted to training/test sets
            proportionally. Labels for each group must be an int greater
            than zero. Must be of size equal to the number of samples, with
            further restrictions on the number of groups.

        verbose : boolean
            A switch indicating whether the fitting should print out messages
            displaying progress. Utilizes tqdm to indicate progress on
            bootstraps.
        """
        # perform checks
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'],
                         y_numeric=True, multi_output=True)
        # preprocess data
        super(AbstractUoILinearClassifier, self).fit(X, y, stratify=stratify,
                                                     verbose=verbose)

        self._fit_intercept(X, y)
        return self
