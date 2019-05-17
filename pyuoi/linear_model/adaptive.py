import numpy as np

def adaptive_estimation_metric(self, P, X, y):

    n_models, n_features, n_samples = P.shape

    # Number of perturbation to use for sensitivity estimation
    T = 1000

    # Peturbation strength
    tau = 0.5

    # Explore the region between no penalization on model size (r^2)
    # and AIC penalty (lambda = 2)
    Lambda = np.logspace(0, 2, 20)

    M_hat = np.zeros((Lambda.size, n_features))

    # Estimate the variance of the additive noise in the linear model
    # Do so from the full model
    sigma_squared = np.norm((y - np.linalg.inv(X.T @ X) @ X.T y))**2/(n_samples - n_features)

    adaptive_loss = np.zeros(Lambda.size)
    g = np.zeros(Lambda_size)

    # For each lambda, find the model that scores the best using the penalty
    # defined by that lambda.Then, calculate the model sensitivity using 
    # random perturbation
    for i, l in enumerate(Lambda):
        loss = [loss(P[j, ...], y, l, sigma_squared) for j in n_models]
        min_loss_idx = np.argmin(loss)
        M_hat[i, :] = P[min_loss_idx, ...] @ y

        Delta = np.random.multivariate_normal(np.zeros(n_samples), 0.25 * np.identity(n_samples),
                            size = T)
        dmudy = np.zeros(n_samples)
        for j, t in enumerate(T): 
            # Perturb the data
            delta = Delta[j, :]
            yy = y + delta

            # Recalculate the loss for all models
            loss = [loss(P[j, ...], y, l, sigma_squared + tau**2) for j in n_models]
            min_loss_idx = np.argmin(loss)
            dmudy += 1/T * np.divide(P[min_loss_idx, ...] @ yy - M_hat[i, :], delta)

        g[i] = 2 * np.sum(dmudy)
        adaptive_loss[i] = (y - M_hat[i, :]).T @ (y - M_hat[i, :]) + g[i] * sigma_squared           

    adaptive_loss_penalty = g[np.argmin(adaptive_loss)]

    # Minimize over lambda to find the best loss
    return adaptive_loss_penalty


def loss(P, y, var, g):
	loss =  (y - P @ y).T @ (y - P @ y) + g * var
	return loss
