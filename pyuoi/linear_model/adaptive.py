import numpy as np
import itertools
import pdb
import time

def adaptive_estimation_penalty(P, X, y):

    # For testing purposes, sigma is fixed and given:
    sigma_squared = 874.1684672295796

    n_models, n_features, n_samples = P.shape

    # Number of perturbation to use for sensitivity estimation
    T = 500

    # Peturbation strength
    tau = 0.5

    # Explore the region between no penalization on model size (r^2)
    # and AIC penalty (lambda = 2)
    Lambda = np.logspace(0, 2, 1)

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
        loss = [loss_fn(y, X @ P[j, ...] @ y, l, sigma_squared) for j in range(n_models)]
        min_loss_idx = np.argmin(loss)
        M_hat[i, :] = (P[min_loss_idx, ...] @ y).ravel()

        Delta = np.random.multivariate_normal(np.zeros(n_samples), 0.25 * np.identity(n_samples),
                            size = T)
        dmudy = np.zeros(n_samples)
        for j in range(T): 
            # Perturb the data
            delta = Delta[j, :]
            yy = y.ravel() + delta

            # Recalculate the loss for all models
            loss = [loss_fn(y, X @ P[j, ...] @ y, l, sigma_squared + tau**2) for j in range(n_models)]
            min_loss_idx = np.argmin(loss)
            dmudy += 1/T * np.divide(X @ P[min_loss_idx, ...] @ yy - X @ M_hat[i, :], delta)

        g[i] = 2 * np.sum(dmudy)
        adaptive_loss[i] = loss_fn(y, X @ M_hat[i, :], sigma_squared, g[i])           
        print(time.time() - t0)

    adaptive_loss_penalty = g[np.argmin(adaptive_loss)]

    # Minimize over lambda to find the best loss
    return adaptive_loss_penalty

def adaptively_score_models(models, y_true, X, g_lambda):

    # For testing purposes, sigma is fixed and given:
    sigma_squared = 874.1684672295796

    scores = np.zeros(models.shape[:-1])

    # Iterate over all but the last axis, which is assumed to be the model
    # coefficients.
    idxs = itertools.product(*[np.arange(l) for l in models.shape[:-1]])
    for idx in idxs:
        pdb.set_trace()
        scores[idx] = loss_fn(y_true, X @ models[idx, :], sigma_squared, g_lambda)

    return scores

def loss_fn(y, estimate, var, g):
    loss =  np.linalg.norm(y - estimate)**2 + g * var
    return loss
