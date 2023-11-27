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
    gpr_list = [GaussianProcessRegressor() for _ in range(n_f)]

    
    # Initialize samples
    X_sample = X_init
    for i, (Y_sample, gpr) in enumerate(zip(Y_sample_multi, gpr_list)):
        gpr.fit(X_sample, Y_sample)
        gpr.optim_np()

    # normalization
    bounds_normal = np.array([[0.0, 1.0]] * dim)

    # # define ref_point according to Appendix A.2
    # min_values = np.min(Y_sample_multi, axis=1)
    # max_values = np.max(Y_sample_multi, axis=1)
    # ref_point = (min_values + max_values) / 2
    # choose the worst point first instead of above method
    ref_point = np.min(Y_sample_multi, axis=1, keepdims=True).squeeze()


    
    # optimize loop
    for i in range(n_init, n_iter+n_init):
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

        Y_next = f_multi(X_next if dim == 1 else X_next.T)

        # Add sample to previous samples
        X_sample = np.vstack((X_sample, X_next))
        Y_sample_list = [np.append(Y, Y_new) for Y, Y_new in zip(Y_sample_list, Y_next)] # not sure about dimension
        
        gpr_list = [GaussianProcessRegressor().fit(X_sample, Y) for Y in Y_sample_list]
        for gpr in gpr_list:
            gpr.optim_np() 
        

    ## TODO: should add a convergence check according to the paper function 24
    ## 计算每次迭代后得到的HV，以进行最终的收敛检查！
    mu_sample = gpr.predict(X_sample)
    pareto = is_non_dominated_np(mu_sample)
    
    return pareto, ref_point