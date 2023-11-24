import jax
import jax.numpy as jnp
from jax import random, vmap, Array, tree_map, lax
import gp
from gp import GP_parameters
from bo import expected_improvement, suggest_next, OptimizerParameters, extend_array
import matplotlib.pyplot as plt
import numpy as np

def plot_approximation(bounds, X_sample, Y_sample, X_next, ls, amp, noise):
    mu, std = gp.predict(bounds, X_sample, Y_sample, l=ls, sigma_f=amp, sigma_y=noise,return_std=True, return_cov=False)
    plt.fill_between(bounds.ravel(), 
                     mu.ravel() + 1.96 * std, 
                     mu.ravel() - 1.96 * std, 
                     alpha=0.1) 
    plt.plot(bounds, mu, 'b-', lw=1, label='Surrogate function')
    plt.plot(X_sample, Y_sample, 'kx', mew=3, label='Noisy samples')
    plt.axvline(x=X_next, ls='--', c='k', lw=1)
    plt.legend()


def optim_process(
    f,
    constrains,
    noise,
    n_init = 5, # Number of initial evaluations before suggesting optimized samples.
    seed = 42,
    dim = 1,
    n = 10, # Number of sampling iterations
    num_steps = 40,
    lr = 0.01,
    plot_figure = False,
):
    key = random.PRNGKey(seed)
    bounds = jnp.asarray(list(constrains.values()))
    # generate data
    X_init = random.uniform(
            key,
            shape=(n_init, dim),
            minval=bounds[:, 0].ravel(),
            maxval=bounds[:, 1].ravel(),
        )

    Y_init = vmap(f)(*X_init.T)

    # Initial settings
    params = GP_parameters(
        lengthscale=jnp.zeros((1, 1)),
        amplitude=jnp.zeros((1, 1)),
    )
    momentums = tree_map(lambda x: x * 0, params)
    scales = tree_map(lambda x: x * 0 + 1, params)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
    # Initialize samples
    X = X_init
    Y = Y_init

    # Expand the array with the same last values, since JAX does not do inplace updates except after being compiled.
    X = extend_array(X, n, 0)
    Y = extend_array(Y, n, 0)

    if plot_figure:
        plt.figure(figsize=(12, n * 3))
        plt.subplots_adjust(hspace=0.4)

    # optimize loop
    for idx in range(n_init, n + n_init):
        # optimize hyper-parameters of GP
        params, momentums, scales  = gp.optimize_mll(params, X, Y, momentums, scales, noise, num_steps, lr)
        momentums = tree_map(lambda x: x * 0, params)
        scales = tree_map(lambda x: x * 0 + 1, params)  
        l_opt, sigma_f_opt = params.lengthscale, params.amplitude

        # Obtain next sampling point from the acquisition function(expected_improvement)
        next_X, key = suggest_next(
            key, 
            X, 
            Y, 
            bounds, 
            ls=l_opt, 
            sigma_f=sigma_f_opt, 
            noise=noise, 
            n_seed=1000, 
            lr=lr, 
            n_epochs=150)

        # visualization for surrogate function, samples and acquisition fucntion
        if plot_figure:
            plt.subplot(n, 2, 2 * (idx-n_init) + 1)
            X_s = np.arange(bounds[:, 0], bounds[:, 1], 0.01).reshape(-1, 1)
            plot_approximation(X_s, X, Y, next_X, l_opt, sigma_f_opt, noise)
            plt.title(f'Iteration {idx-n_init+1}')

        # add the new point into samples
        X = X.at[idx, ...].set(next_X)
        Y = Y.at[idx].set(f(*next_X))

    best_target = float(Y.max())
    best_params = {k: v for (k, v) in zip(constrains.keys(), X[Y.argmax()])}

    optimizer_params = OptimizerParameters(
        target=best_target, params=best_params, f=f, params_all=X, target_all=Y
    )

    return optimizer_params