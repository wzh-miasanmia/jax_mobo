import matplotlib.pyplot as plt
import numpy as np
from GP_class import GaussianProcessRegressor
from sobo import propose_location, expected_improvement
from typing import Callable, NamedTuple, Tuple, Union, Optional
from simulation import select_para_and_simulation

class OptimizerParameters(NamedTuple):
    """
    Object holding the results of the optimization.
    """
    target: Union[np.ndarray, float]
    params: np.ndarray
    params_all: np.ndarray
    target_all: np.ndarray

def plot_approximation_1d(bounds, X_sample, Y_sample, X_next, gpr):
    mu, std = gpr.predict(bounds, return_std=True)
    plt.fill_between(bounds.ravel(), 
                     mu.ravel() + 1.96 * std, 
                     mu.ravel() - 1.96 * std, 
                     alpha=0.1) 
    plt.plot(bounds, mu, 'b-', lw=1, label='Surrogate function')
    plt.plot(X_sample, Y_sample, 'kx', mew=3, label='Noisy samples')
    plt.axvline(x=X_next, ls='--', c='k', lw=1)
    plt.legend()

def normalize(X, bounds):
    return (X - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])

def unnormalize(X, bounds):
    return X * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]

def optim_process(
    scheme_setup,
    noise,
    SCHEME = "teno6",
    acq = expected_improvement,
    n_init = 5, # Number of initial evaluations before suggesting optimized samples.
    n_iter = 20, # Number of sampling iterations
    normalization = False,
    DEBUG = True,
):
    # transform setup into constraints and bounds
    constraints = {}
    for key, value in scheme_setup.items():
        for i in range(1, (len(value) // 2 + 1)):
            para_name_key = f"para{i}_name"
            para_range_key = f"para{i}_range"
            para_name = value[para_name_key]
            para_range = value[para_range_key]
            constraints[para_name] = tuple(para_range)

    bounds = np.asarray(list(constraints.values()))

    # GP model initialization
    gpr = GaussianProcessRegressor(sigma_y=noise)
    dim = len(bounds)

    # generate data
    X_init = np.random.uniform(
        low=bounds[:, 0].ravel(),
        high=bounds[:, 1].ravel(),
        size=(n_init, dim),
    )
    X = X_init
    X_sample = np.round(X, 4)

    # jax-fluid simulation
    Y_sample = np.array([select_para_and_simulation(para=p, N_SAMPLES=n_init, SCHEME=SCHEME, scheme_setup=scheme_setup, DEBUG=DEBUG) for p in X_sample]) 
    
    gpr.fit(X_sample, Y_sample)
    gpr.optim_np()

    if dim == 1:
        plt.figure(figsize=(12, n_iter * 3))
        plt.subplots_adjust(hspace=0.4)


    # normalization
    bounds_normal = np.array([[0.0, 1.0]] * dim)

    # optimize loop
    for i in range(n_init, n_iter+n_init):
        if normalization:
            # normalization
            X_sample_normal = normalize(X_sample, bounds)
            # Obtain next sampling point from the acquisition function(expected_improvement)
            X_next_normal = propose_location(acq, X_sample_normal, Y_sample, gpr, bounds_normal)
            # de-normalization
            X_next = unnormalize(X_next_normal, bounds)
        else:
            # Obtain next sampling point from the acquisition function(expected_improvement)
            X_next = propose_location(acq, X_sample, Y_sample, gpr, bounds)
        
        X_next = np.round(X_next, 4)

        # put into CFD simulation
        if dim == 1:
            Y_next = select_para_and_simulation(para=X_next, N_SAMPLES=n_init, SCHEME=SCHEME, scheme_setup=scheme_setup, DEBUG=DEBUG)
        else:
            Y_next = select_para_and_simulation(para=X_next.T, N_SAMPLES=n_init, SCHEME=SCHEME, scheme_setup=scheme_setup, DEBUG=DEBUG)

        
        # visualization for surrogate function, samples and acquisition fucntion
        if dim == 1:
            plt.subplot(n_iter, 2, 2 * (i-n_init) + 1)
            X_s = np.arange(bounds[:, 0], bounds[:, 1], 0.01).reshape(-1, 1)
            plot_approximation_1d(X_s, X_sample, Y_sample, X_next, gpr)
            plt.title(f'Iteration {i-n_init+1}')

        # Add sample to previous samples
        X_sample = np.vstack((X_sample, X_next))
        Y_sample = np.append(Y_sample, Y_next)
        gpr.fit(X_sample, Y_sample)
        gpr.optim_np()

    best_target = float(Y_sample.max())
    best_params = {k: v for (k, v) in zip(constraints.keys(), X_sample[Y_sample.argmax()])}

    optimizer_params = OptimizerParameters(
        target=best_target, params=best_params, params_all=X_sample, target_all=Y_sample
    )

    return optimizer_params