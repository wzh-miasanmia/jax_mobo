from GP_class import GaussianProcessRegressor
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from math import inf
import numpy as np
from scipy.stats import norm
from pareto_cal import is_non_dominated_np
from HV_cal import Hypervolume, plot_pareto_hv

def expected_hypervolume_improvement(X, X_sample, Y_sample_multi, gpr_list, ref_point):
    '''
    Computes the EHVI at points X based on existing samples X_sample
    and Y_sample using a list of Gaussian process surrogate models.
    
    Args:
        X: Points at which EHVI shall be computed (m x d).
        X_sample: Sample locations (n x d).
        Y_sample_multi: Sample values (n x f).
        gpr_list: A list of GaussianProcessRegressor fitted to samples.
        ref_point: Reference point to calculate hypervolume
    Returns:
        Expected improvements at points X.
    '''
    # definition of hypervolume
    HV = Hypervolume(ref_point)
    
    # previous pareto and hypervolume
    mu_sample_list = [gpr.predict(X_sample) for gpr in gpr_list]
    previous_points = np.column_stack(mu_sample_list)
    previous_pareto_mask = is_non_dominated_np(previous_points)
    previous_pareto = previous_points[previous_pareto_mask]
    HV_previous = HV.compute(previous_pareto)
    
    # new pareto and hypervolume
    mu_list, sigma_list = zip(*[gpr.predict(X, return_std=True) for gpr in gpr_list])
    new_point = np.column_stack(mu_list)
    new_points = np.vstack((new_point, previous_points))
    new_pareto_mask = is_non_dominated_np(new_points)
    new_pareto = new_points[new_pareto_mask]
    HV_new = HV.compute(new_pareto)
    
    # calculate ehvi
    improvement = HV_new - HV_previous
    Z = improvement / np.column_stack(sigma_list)
    ehvi = improvement * norm.cdf(Z) + np.column_stack(sigma_list) * norm.pdf(Z)
    ehvi[np.column_stack(sigma_list) == 0.0] = 0.0 # MUST BE a scalar value, need to consider

    return ehvi

def propose_location(acquisition, X_sample, Y_sample_multi, gpr_list, bounds, ref_point, n_restarts=25):
    '''
    Proposes the next sampling point by optimizing the acquisition function.
    
    Args:
        acquisition: Acquisition function.
        X_sample: Sample locations (n x d).
        Y_sample_multi: Sample values (n x f).
        gpr_list: A list of GaussianProcessRegressor fitted to samples.

    Returns:
        Location of the acquisition function maximum.
    '''
    dim = X_sample.shape[1]
    min_val = 1
    min_x = None
    
    def min_obj(X):
        # Minimization objective is the negative acquisition function
        return -acquisition(X.reshape(-1, dim), X_sample, Y_sample_multi, gpr_list, ref_point).flatten()
    
    # Find the best optimum by starting from n_restart different random points.
    for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, dim)):
        res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')        
        if res.fun < min_val:
            min_val = res.fun
            min_x = res.x           
            
    return min_x
