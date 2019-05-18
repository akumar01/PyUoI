import numpy as np
import itertools
import pdb
import time
from scipy.sparse.linalg import lsqr

def adaptive_estimation_penalty(P, supports, X, y):

    # For testing purposes, sigma is fixed and given:
    sigma_squared = 874.1684672295796

    n_models, n_features, n_samples = P.shape

    # Number of perturbation to use for sensitivity estimation
    T = 500

    # Peturbation strength
    tau = 0.5 * np.sqrt(sigma_squared)

    # Explore the region between no penalization on model size (r^2)
    # and AIC penalty (lambda = 2)
    Lambda = np.logspace(np.log10(0.05), np.log10(np.log(n_samples)), 20)

    M_hat = np.zeros((Lambda.size, n_features))

    # Estimate the variance of the additive noise in the linear model
    # Do so from the full model
    # sigma_squared = np.norm((y - np.linalg.inv(X.T @ X) @ X.T y))**2/(n_samples - n_features)

    adaptive_loss = np.zeros(Lambda.size)
    g = np.zeros(Lambda.size)

    # For each lambda, find the model that scores the best using the penalty
    # defined by that lambda.Then, calculate the model sensitivity using 
    # random perturbation
    for i, l in enumerate(Lambda):
        t0 = time.time()
        loss = [loss_fn(y, X @ P[j, ...] @ y, sigma_squared, 
                np.count_nonzero(1 * supports[j, :]), l) for j in range(n_models)]
        min_loss_idx = np.argmin(loss)
        M_hat[i, :] = (P[min_loss_idx, ...] @ y).ravel()

        Delta = np.random.multivariate_normal(np.zeros(n_samples), tau**2 * np.identity(n_samples),
                            size = T)
        
        # Resulting estimators when the data has been perturbed
        perturbation_responses = np.zeros((T, n_samples))

        for j in range(T): 
            # Perturb the data
            delta = Delta[j, :]
            yy = y.ravel() + delta

            # Recalculate the loss for all models
            loss = [loss_fn(yy, X @ P[k, ...] @ yy, sigma_squared + tau**2, 
                    np.count_nonzero(1 * supports[k, :]), l) for k in range(n_models)]
            min_loss_idx = np.argmin(loss)
            perturbation_responses[j, :] = X @ P[min_loss_idx, ...] @ yy - X @ M_hat[i, :]

            # dmudy += 1/T * np.divide(X @ P[min_loss_idx, ...] @ yy - X @ M_hat[i, :], delta)

        # For each component, estimate that component of the derivative as the regression coefficient
        # of a linear regression problem
        dmudy = np.array([lsqr(Delta[:, j, np.newaxis], perturbation_responses[:, j])[0][0] for j in range(n_samples)])
        g[i] = np.sum(dmudy)
        adaptive_loss[i] = loss_fn(y, X @ M_hat[i, :], sigma_squared, 1, g[i])           
        print(time.time() - t0)

    adaptive_loss_penalty = Lambda[np.argmin(adaptive_loss)]
    # Minimize over lambda to find the best loss
    return adaptive_loss_penalty

def adaptively_score_models(models, y_true, X, lambda_hat):

    # For testing purposes, sigma is fixed and given:
    sigma_squared = 874.1684672295796

    scores = np.zeros(models.shape[:-1])

    # Iterate over all but the last axis, which is assumed to be the model
    # coefficients.
    idxs = itertools.product(*[np.arange(l) for l in models.shape[:-1]])
    for idx in idxs:
        scores[idx[0], idx[1]] = loss_fn(y_true, X @ models[idx[0], idx[1], :], sigma_squared, 
                                        np.count_nonzero(models[idx[0], idx[1], :]), lambda_hat)
    return -1*scores

def loss_fn(y, estimate, var, M, g):
    loss =  np.linalg.norm(y - estimate)**2 + g * M * var
    return loss
