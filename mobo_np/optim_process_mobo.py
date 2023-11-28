import matplotlib.pyplot as plt
import numpy as np
from GP_class import GaussianProcessRegressor
from mobo import propose_location, expected_hypervolume_improvement
from pareto_cal import is_non_dominated_np
from HV_cal import Hypervolume, plot_pareto_hv

def normalize(X, bounds):
    return (X - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])

def unnormalize(X, bounds):
    return X * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]

def from_X_calculate_HV(X_sample, gpr_list, ref_point):
    mu_sample_list = []
    for gpr in gpr_list:
        mu_sample = gpr.predict(X_sample).reshape(-1) # (n_samples,)
        mu_sample_list.append(mu_sample)
    mu_samples = (np.vstack(mu_sample_list)).T # (n_samples, n_functions)
    pareto_mask = is_non_dominated_np(mu_samples)
    pareto = mu_samples[pareto_mask]    
    HV_class = Hypervolume(ref_point)
    HV = HV_class.compute(pareto)
    return HV
    
    
def optim_process(
    f_multi, # list of functions
    constrains,
    noise,
    acq = expected_hypervolume_improvement,
    n_init = 5, # Number of initial evaluations before suggesting optimized samples.
    n_iter = 10, # Number of sampling iterations
    normalization = False,
):

    bounds = np.asarray(list(constrains.values()))
    dim = len(bounds)
    # generate data
    X_init = np.random.uniform(
        low=bounds[:, 0].ravel(),
        high=bounds[:, 1].ravel(),
        size=(n_init, dim),
    )
    X = X_init
    Y_sample_multi = f_multi(X if dim == 1 else X.T) # [n_f, n_samples, 1]
    n_f = Y_sample_multi.shape[0] # number of functions
    gpr_list = [GaussianProcessRegressor(sigma_y=noise) for _ in range(n_f)]

    # Initialize samples
    X_sample = X_init
    for _, (Y_sample, gpr) in enumerate(zip(Y_sample_multi, gpr_list)):
        gpr.fit(X_sample, Y_sample)
        gpr.optim_np()

    # normalization
    bounds_normal = np.array([[0.0, 1.0]] * dim)

    # define ref_point according to Appendix A.2
    min_values = np.min(Y_sample_multi, axis=1)
    max_values = np.max(Y_sample_multi, axis=1)
    ref_point = ((min_values + max_values) / 2).squeeze()
    # choose the worst point instead
    # ref_point = np.min(Y_sample_multi, axis=1, keepdims=True).squeeze()

    # inisialize a HV list to save Hypervolume of each iteration
    HV_list = []
    HV_RelErr_list = []
    # optimize loop
    for i in range(n_iter):
        if normalization:
            # normalization
            X_sample_normal = normalize(X_sample, bounds)
            # Obtain next sampling point from the acquisition function(expected_improvement)
            X_next_normal = propose_location(acq, X_sample_normal, Y_sample_multi, gpr_list, bounds_normal, ref_point)
            # de-normalization
            X_next = unnormalize(X_next_normal, bounds)
        else:
            # Obtain next sampling point from the acquisition function(expected_improvement)
            X_next = propose_location(acq, X_sample, Y_sample_multi, gpr_list, bounds, ref_point)

        # Add sample to previous samples
        X_sample = np.vstack((X_sample, X_next))
        # calculate HV values of this iteration and add it to the list
        HV = from_X_calculate_HV(X_sample, gpr_list, ref_point)
        HV_list.append(HV)
        # convergence criteria
        N = 10
        if i > N:
            # Calculate HVi_SM
            HVi_SM = sum(HV_list[i - j] for j in range(N)) / N
            # Calculate HV RelErr
            HV_RelErr = (HV - HVi_SM) / HVi_SM
            HV_RelErr_list.append(HV_RelErr)
        # use the real model to calculate new Y and fit the GP model again
        # TODO: change this to numerical simulation
        Y_next = f_multi(X_next if dim == 1 else X_next.T)         
        Y_sample_multi = [np.append(Y, Y_new) for Y, Y_new in zip(Y_sample_multi, Y_next)]

        # update GP model parameters
        for _, (Y_sample, gpr) in enumerate(zip(Y_sample_multi, gpr_list)):
            gpr.fit(X_sample, Y_sample)
            gpr.optim_np()
            
    ## TODO: add a convergence check according to the paper function 24
    mu_sample_list = []
    for gpr in gpr_list:
        mu_sample_list.append(gpr.predict(X_sample))
    mu_sample = (np.vstack(mu_sample_list)).T # (n_samples, n_functions)
    pareto_mask = is_non_dominated_np(mu_sample) 
    pareto = mu_sample[pareto_mask]
    return pareto, HV_RelErr_list # TODO: return the X_sample that corresponds to pareto