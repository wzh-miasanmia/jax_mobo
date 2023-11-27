from GP_class import GaussianProcessRegressor
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from math import inf
import numpy as np
from scipy.stats import norm
from pareto_cal import is_non_dominated_np


def expected_improvement(X, X_sample, Y_sample, gpr:GaussianProcessRegressor):
    '''
    Computes the EI at points X based on existing samples X_sample
    and Y_sample using a Gaussian process surrogate model.
    
    Args:
        X: Points at which EI shall be computed (m x d).
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x f).
        gpr: A GaussianProcessRegressor fitted to samples. An instance of the GaussianProcessRegressor class from scikit-learn, already fitted to the X_sample and Y_sample.
        xi: Exploitation-exploration trade-off parameter.
    
    Returns:
        Expected improvements at points X.
    '''
    mu, sigma = gpr.predict(X, return_std=True)
    mu_sample = gpr.predict(X_sample)


    new_points = np.stack(mu, mu_sample)
    new_pareto = is_non_dominated_np(new_points)
    
    previous_points = mu_sample - sigma
    previous_pareto = is_non_dominated_np(previous_points)
    
    ## Todo: definie HV
    HV = 1
    
    improvement = HV(previous_pareto) - HV(previous_points)
    Z = improvement / sigma
    ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma == 0.0] = 0.0

    return ei

def propose_location(acquisition, X_sample, Y_sample, gpr, bounds, n_restarts=25):
    '''
    Proposes the next sampling point by optimizing the acquisition function.
    
    Args:
        acquisition: Acquisition function.
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        gpr: A GaussianProcessRegressor fitted to samples.

    Returns:
        Location of the acquisition function maximum.
    '''
    dim = X_sample.shape[1]
    min_val = 1
    min_x = None
    
    def min_obj(X):
        # Minimization objective is the negative acquisition function
        return -acquisition(X.reshape(-1, dim), X_sample, Y_sample, gpr).flatten()
    
    # Find the best optimum by starting from n_restart different random points.
    for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, dim)):
        res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')        
        if res.fun < min_val:
            min_val = res.fun
            min_x = res.x           
            
    return min_x