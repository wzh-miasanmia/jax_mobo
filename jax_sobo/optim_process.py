import jax
import jax.numpy as jnp
from jax import random, vmap, Array
import gp
from bo import expected_improvement, suggest_next, OptimizerParameters

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
):
    key = random.PRNGKey(seed)
    bounds = jnp.asarray(list(constrains.values()))
    # generate data
    X_init = random.uniform(
            key,
            shape=(n_init, dim),
            minval=bounds[:, 0],
            maxval=bounds[:, 1],
        )

    Y_init = vmap(f)(*X_init.T)

    # Initial settings
    # Number of sampling iterations.

    theta = jnp.array([1.0,1.0])

    # Initialize samples
    X = X_init
    Y = Y_init

    # optimize loop
    for idx in range(n_init, n + n_init):
        # optimize hyper-parameters of GP
        theta = gp.optimize_mll(theta, X, Y, noise, num_steps, lr, method='SGD')
        l_opt, sigma_f_opt = theta

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
        X = X.at[idx, ...].set(next_X)
        Y = Y.at[idx].set(f(*next_X))

    best_target = float(Y.max())
    best_params = {k: v for (k, v) in zip(constrains.keys(), X[Y.argmax()])}

    optimizer_params = OptimizerParameters(
            target=best_target, params=best_params, f=f, params_all=X, target_all=Y
        )

    return optimizer_params