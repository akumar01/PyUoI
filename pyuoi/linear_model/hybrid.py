import numpy as np

from .base import AbstractUoILinearRegressor

from sklearn.linear_model import LinearRegression
from sklearn.linear_model.coordinate_descent import _alpha_grid
from sklearn.linear_model import ElasticNet, Lasso
import pdb

from sklearn.linear_model.base import (
    LinearModel, _preprocess_data, SparseCoefMixin)
from sklearn.metrics import r2_score, accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.utils import check_X_y


from pyuoi import utils
from pyuoi.mpi_utils import get_chunk_size, get_buffer_mask

from .utils import stability_selection_to_threshold, intersection



class UoI_Hybrid(AbstractUoILinearRegressor):

    def __init__(self, n_lambdas=48, alphas=np.array([0.5]),
                 n_boots_sel=48, n_boots_est=48, selection_frac=0.9,
                 estimation_frac=0.9, stability_selection=1.,
                 estimation_score='r2', warm_start=True, eps=1e-3,
                 copy_X=True, fit_intercept=True, normalize=True,
                 random_state=None, max_iter=1000,
                 comm=None):
        super(UoI_Hybrid, self).__init__(
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
        self.__selection_lm1 = ElasticNet(
            fit_intercept=fit_intercept,
            normalize=normalize,
            max_iter=max_iter,
            copy_X=copy_X,
            warm_start=warm_start,
            random_state=random_state)
        self.__selection_lm2 = Lasso(
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
        pass

    @property
    def selection_lm1(self):
        return self.__selection_lm1

    @property
    def selection_lm2(self):
        return self.__selection_lm2


    # Implement a custom fitting command to accommodate side by side comparison of
    # using Lasso and ElasticNet as selection_lms

    def fit(self, X, y, stratify=None, verbose=False):
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
        self.reg_params_en_ = self.get_reg_params_en(X, y)
        self.n_reg_params_en_ = len(self.reg_params_en_)

        self.reg_params_lasso_ = self.get_reg_params_lasso(X, y)
        self.n_reg_params_lasso_ = len(self.reg_params_lasso_)

        rank = 0
        size = 1
        if self.comm is not None:
            rank = self.comm.rank
            size = self.comm.size
        chunk_size, buf_len = get_chunk_size(rank, size, self.n_boots_sel)

        # initialize selection for ElasticNet
        selection_coefs_en = np.zeros((buf_len, self.n_reg_params_en_, n_coef),
                                   dtype=np.float32)
        # initialize selection for Lasso
        selection_coefs_lasso = np.zeros((buf_len, self.n_reg_params_lasso_, n_coef),
                                   dtype=np.float32)

        # iterate over bootstraps
        for bootstrap in range(chunk_size):

            # reset the coef between bootstraps for 1st selection lm (ElasticNet)
            if hasattr(self.selection_lm1, 'coef_'):
                self.selection_lm1.coef_ = np.zeros_like(
                    self.selection_lm1.coef_,
                    dtype=X.dtype,
                    order='F')

            # reset the coef between bootstraps for 2nd selection lm (Lasso)
            if hasattr(self.selection_lm2, 'coef_'):
                self.selection_lm2.coef_ = np.zeros_like(
                    self.selection_lm2.coef_,
                    dtype=X.dtype,
                    order='F')

            # draw a resampled bootstrap --> keep the same between ElasticNet and Lasso
            X_rep, X_test, y_rep, y_test = train_test_split(
                X, y,
                test_size=1 - self.selection_frac,
                stratify=stratify,
                random_state=self.random_state
            )


            # Fit over all regularization parameters for ElasticNet
            for reg_param_idx, reg_params in enumerate(self.reg_params_en_):
                # reset the regularization parameter
                self.selection_lm1.set_params(**reg_params)
                # rerun fit
                self.selection_lm1.fit(X_rep, y_rep)
                # store coefficients
                selection_coefs_en[bootstrap, reg_param_idx, :] = \
                    self.selection_lm1.coef_.ravel()

            # Fit over all regularization parameters for Lasso
            for reg_param_idx, reg_params in enumerate(self.reg_params_lasso_):
                # reset the regularization parameter
                self.selection_lm2.set_params(**reg_params)
                # rerun fit
                self.selection_lm2.fit(X_rep, y_rep)
                # store coefficients
                selection_coefs_lasso[bootstrap, reg_param_idx, :] = \
                    self.selection_lm2.coef_.ravel()

        # Ignore the block of code about self.comm, modify only the else statement

        # if distributed, gather selection coefficients to 0,
        # perform intersection, and broadcast results
        if self.comm is not None:
            self.comm.Barrier()
            recv = None
            if rank == 0:
                recv = np.zeros((buf_len * size, self.n_reg_params_, n_coef),
                                dtype=np.float32)
            self.comm.Gather(selection_coefs, recv, root=0)
            supports = None
            shape = None
            if rank == 0:
                mask = get_buffer_mask(size, self.n_boots_sel)
                recv = recv[mask]
                supports = self.intersect(recv, self.selection_thresholds_)
                shape = supports.shape
            shape = self.comm.bcast(shape, root=0)
            if rank != 0:
                supports = np.zeros(shape, dtype=np.float32)
            supports = self.comm.bcast(supports, root=0)
            self.supports_ = supports
        else:
            self.supports_en_ = self.intersect(selection_coefs_en,
                                            self.selection_thresholds_)
            self.supports_lasso_ = self.intersect(selection_coefs_lasso,
                                            self.selection_thresholds_)

        self.n_supports_en_ = self.supports_en_.shape[0]
        self.n_supports_lasso_ = self.supports_lasso_.shape[0]

        pdb.set_trace()
        #####################
        # Estimation Module #
        #####################
        # set up data arrays

        chunk_size, buf_len = get_chunk_size(rank, size, self.n_boots_est)

        # coef_ for each bootstrap for each support
        self.estimates_ = np.zeros((buf_len, self.n_supports_, n_coef),
                                   dtype=np.float32)

        # score (r2/AIC/AICc/BIC) for each bootstrap for each support
        self.scores_ = np.zeros((buf_len, self.n_supports_), dtype=np.float32)

        n_tile = n_coef // n_features
        # iterate over bootstrap samples
        for bootstrap in range(chunk_size):

            # draw a resampled bootstrap
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=1 - self.estimation_frac,
                stratify=stratify,
                random_state=self.random_state
            )

            # iterate over the regularization parameters
            for supp_idx, support in enumerate(self.supports_):
                # extract current support set
                # if nothing was selected, we won't bother running OLS
                if np.any(support):
                    # compute ols estimate
                    self.estimation_lm.fit(X_train[:, support], y_train)

                    # store the fitted coefficients
                    self.estimates_[
                        bootstrap, supp_idx, np.tile(support, n_tile)] = \
                        self.estimation_lm.coef_.ravel()

                    # obtain predictions for scoring
                    y_pred = self.estimation_lm.predict(X_test[:, support])
                else:
                    # no prediction since nothing was selected
                    y_pred = np.zeros(y_test.size)

                # calculate estimation score
                self.scores_[bootstrap, supp_idx] = self.score_predictions(
                    self.estimation_score, y_test, y_pred, support)

        if self.comm is not None:
            self.comm.Barrier()
            est_recv = None
            scores_recv = None
            self.rp_max_idx_ = None
            best_estimates = None
            if rank == 0:
                est_recv = np.zeros((buf_len * size, self.n_supports_, n_coef),
                                    dtype=np.float32)
                scores_recv = np.zeros((buf_len * size, self.n_supports_),
                                       dtype=np.float32)
            self.comm.Gather(self.scores_, scores_recv, root=0)
            self.comm.Gather(self.estimates_, est_recv, root=0)
            if rank == 0:
                mask = get_buffer_mask(size, self.n_boots_est)
                self.estimates_ = est_recv[mask]
                self.scores_ = scores_recv[mask]
                self.rp_max_idx_ = np.argmax(self.scores_, axis=1)
                best_estimates = self.estimates_[np.arange(self.n_boots_est),
                                                 self.rp_max_idx_, :]
            self.rp_max_idx_ = self.comm.bcast(self.rp_max_idx_, root=0)
            best_estimates = self.comm.bcast(best_estimates, root=0)
        else:
            self.rp_max_idx_ = np.argmax(self.scores_, axis=1)
            # extract the estimates over bootstraps from model with best
            # regularization parameter value
            best_estimates = self.estimates_[np.arange(self.n_boots_est),
                                             self.rp_max_idx_, :]

        # take the median across estimates for the final, bagged estimate
        self.coef_ = (np.median(best_estimates, axis=0)
                      .reshape(n_tile, n_features))

        return self

    def get_reg_params(self):
        pass


    def get_reg_params_en(self, X, y):
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

    def get_reg_params_lasso(self, X, y):
        alphas = _alpha_grid(
            X=X, y=y,
            l1_ratio=1.0,
            fit_intercept=self.fit_intercept,
            eps=self.eps,
            n_alphas=self.n_lambdas,
            normalize=self.normalize
        )
        return [{'alpha': a} for a in alphas]
