import numpy as np
from pyuoi.utils import log_likelihood_glm, MIC
    
### MAKE SURE TO CHECK THAT INTERCEPTS ARE NEGLIGIBLE THROUGHOUT

def score_predictions(y, y_pred, n_features, penalty):
    ll = log_likelihood_glm('normal', y, y_pred)
    score = MIC(ll, n_features, penalty)
    return score

# Attempt 1: Shouldn't the GDF of OLS just be the number of 
# features? This makes the whole procedure very straightforward
def naive_adaptive_penalty(X, y, estimates, support_idxs, supports, lambdas):

    # Perform a minimization over lambda
    estimator_losses = np.zeros(lambdas.size)

    for i, l in enumerate(lambdas):

        y_pred = X @ estimates[support_idxs[i], :]
        gdf = np.count_nonzero(estimates[support_idxs[i], :])
        estimator_losses[i] = gdf - log_likelihood_glm('normal', y, y_pred)

    lambda_hat = lambdas[np.argmin(estimator_losses)]
    return lambda_hat

# calculate the adaptive mdoel penalty 
def calc_adaptive_penalty(P, supports, X, y):

    # First step: Over the range of lambda values considered, and the set of 
    # models, calculate the 

    y = y.ravel()

    n_models, n_features, n_samples = P.shape

    # Let sigma be known and fixed for now:
    sigma_squared = 1

    # Peturbation strength
    tau = 0.5 * np.sqrt(sigma_squared)

    Lambda = np.linspace(0, 2 * np.log(n_samples), 40)

    M_hat = np.zeros((Lambda.size, n_features))

def adaptive_estimation_penalty(P, supports, X, y):

    y = y.ravel()

    n_models, n_features, n_samples = P.shape

    # Let sigma be known and fixed for now:
    sigma_squared = 1

    # Number of perturbation to use for sensitivity estimation
    T = n_samples

    # Peturbation strength
    tau = 0.5 * np.sqrt(sigma_squared)

    Lambda = np.linspace(0, np.log(n_samples), 40)

    M_hat = np.zeros((Lambda.size, n_features))

    loss_estimate = np.zeros(Lambda.size)
    D_lambda = np.zeros(Lambda.size)

    # For each lambda, find the model that scores the best using the penalty
    # defined by that lambda.Then, calculate the model sensitivity using 
    # random perturbation
    for i, l in enumerate(Lambda):
        t0 = time.time()
        model_losses = [model_loss(y.ravel(), X @ P[j, ...] @ y, sigma_squared,
                np.count_nonzero(1 * supports[j, :]), l) for j in range(n_models)]

        min_loss_idx = np.argmin(model_losses)
        print(min_loss_idx)
        M_hat[i, :] = (P[min_loss_idx, ...] @ y).ravel()

        y_tilde = np.random.multivariate_normal(y.ravel(), sigma_squared * np.identity(n_samples),
                            size = T)
        
        y_star = np.array([y.ravel() + tau * (y_tilde[j, :] - y.ravel()) for j in range(T)])

        # Resulting estimators when the data has been perturbed
        perturbation_responses = np.zeros((T, n_samples))

        for j in range(T): 
            # Perturb the data
            yy = y_star[j, :]

            # Recalculate the model losses and select the best model estimate for the 
            # perturbed data
            perturbed_model_losses = [model_loss(yy.ravel(), X @ P[k, ...] @ yy, sigma_squared + tau**2, 
                    np.count_nonzero(1 * supports[k, :]), l) for k in range(n_models)]
            min_loss_idx = np.argmin(perturbed_model_losses)
            perturbation_responses[j, :] = X @ P[min_loss_idx, ...] @ yy
            # dmudy += 1/T * np.divide(X @ P[min_loss_idx, ...] @ yy - X @ M_hat[i, :], delta)

        # Estimate the generalized degrees of freedom from means over the Monte Carlo procedure:
        D_lambda[i] = 1/tau**2 * 1/(T - 1) * np.sum(np.multiply(
                                    (perturbation_responses - np.mean(perturbation_responses, axis = 0)), 
                                    (y_star - np.mean(y_star, axis = 0))))

        loss_estimate[i] = loss_estimator_loss(y.ravel(), X @ M_hat[i, :], D_lambda[i])   
        #print(time.time() - t0)

    adaptive_loss_penalty = Lambda[np.argmin(loss_estimate)]
    # Minimize over lambda to find the best loss
    return adaptive_loss_penalty, D_lambda, loss_estimate