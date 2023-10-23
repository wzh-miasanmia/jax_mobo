import jax.numpy as jnp
from jax import grad, jit, lax, tree_util, vmap
from collections import namedtuple
from functools import partial
from jax.scipy.linalg import cholesky, solve_triangular
from typing import Any, Callable, Tuple, Union, Optional
import matplotlib.pyplot as plt

# parameters
GP_parameters = namedtuple("GP_parameters", ["noise", "amplitude", "lengthscale"])

# Convert the input to a nonlinear form
def softplus(x):
    return jnp.logaddexp(x, 0.0)

# covariance matrix calculation
def cov(k: Callable, x1, x2=None):
    if x2 is None:
        cov_matrix = vmap(lambda x: vmap(lambda y: k(x, y))(x1))(x1)
        return cov_matrix
    else:
        cov_matrix = vmap(lambda x: vmap(lambda y: k(x, y))(x1))(x2).T
        return cov_matrix

# use squared exponential kernel(Gaussian kernel, RBF kernel) as k
def gaussian_kernel(x1, x2, ls):
    return jnp.exp(-jnp.sum((x1 - x2) ** 2 / (2 * ls ** 2)))


# GP
def gaussian_process(
    params: GP_parameters, 
    x, 
    y, 
    x_new, 
    compute_ml: bool = False
):
    # Number of points in the prior distribution
    n = x.shape[0]

    # parameters and kernel
    noise, amp, ls = tree_util.tree_map(softplus, params)
    kernel = partial(gaussian_kernel, ls=ls)

    # Covariance matrix K[X,X]
    K = amp * cov(kernel, x)

    # Covariance matrix K[X,X] with noise
    K_y = K + (jnp.eye(n) * (noise + 1e-6))

    # Normalization of measurements
    ymean = jnp.mean(y)
    y = y - ymean

    # Compute K_inv_y
    L = cholesky(K_y, lower=True)
    K_inv_y = solve_triangular(L.T, solve_triangular(L, y, lower=True))

    if compute_ml:
        # Compute the marginal likelihood using its closed form:
        # log(P) = - 0.5 yK^-1y - 0.5 |K-sigmaI| - n/2 log(2pi)
        fitting = y.T.dot(K_inv_y)

        # Compute the determinant using the Lower Diagonal Factorization
        # since it makes it only the diagonal multiplication (sum of logs)
        penalty = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))

        ml = -0.5 * jnp.sum(fitting + penalty + n * jnp.log(2.0 * jnp.pi))

        # Add the amplitude hyperparameter to the marginal likelihood
        ml += 0.5 * jnp.log(2.0 * jnp.pi) + jnp.log(amp.reshape()) ** 2
        return -ml

    # Compute the covariance with x and new point xt
    K_cross = amp * cov(kernel, x, x_new)

    # Compute the covariance with the new point xt
    K_new = amp * cov(kernel, x_new)

    # Return the mean and standard devition of the Gaussian Proceses
    mean = jnp.dot(K_cross.T, K_inv_y) + ymean
    v = solve_triangular(L, K_cross, lower=True)
    var = K_new - jnp.dot(v.T, v)
    # std = jnp.sqrt(var) if n == 1 else jnp.diag(var)

    return mean, var

marginal_likelihood = partial(gaussian_process, compute_ml=True)
grad_fun = jit(grad(marginal_likelihood))
predict = jit(partial(gaussian_process, compute_ml=False))


@jit
def train(
    x,
    y,
    params: GP_parameters,
    momentums: GP_parameters,
    scales: GP_parameters,
    lr: float = 0.01,
    nsteps: int = 20,
) -> Tuple[GP_parameters, GP_parameters, GP_parameters]:
    """
    Training function of the Gaussian Process Regressor.

    Parameters:
    -----------
    x: Sampled points.
    y: Target values of the sampled points.
    params: Hyperparameters of the GP.
    lr: Learning rate of the train step.
    nsteps: Number of epochs to train.

    Returns:
    --------
    Tuple with the trained `params`, `momentums` and `scales`.
    """

    def train_step(
        params: GP_parameters, momentums: GP_parameters, scales: GP_parameters
    ) -> Tuple:
        grads = grad_fun(params, x, y)
        momentums = tree_util.tree_map(
            lambda m, g: 0.9 * m + 0.1 * g, momentums, grads
        )
        scales = tree_util.tree_map(
            lambda s, g: 0.9 * s + 0.1 * g ** 2, scales, grads
        )
        params = tree_util.tree_map(
            lambda p, m, s: p - lr * m / jnp.sqrt(s + 1e-5),
            params,
            momentums,
            scales,
        )
        return params, momentums, scales

    params, momentums, scales = lax.fori_loop(
        0,
        nsteps,
        lambda _, v: train_step(*v),
        (params, momentums, scales),
    )

    return params, momentums, scales


def plot_gp(mu, cov, X, X_train=None, Y_train=None, samples=[], save_path=None):
    X = X.ravel()
    mu = mu.ravel()
    uncertainty = 1.96 * jnp.sqrt(jnp.diag(cov))
    
    plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.1)
    plt.plot(X, mu, label='Mean')
    for i, sample in enumerate(samples):
        plt.plot(X, sample, lw=1, ls='--', label=f'Sample {i+1}')
    if X_train is not None:
        plt.plot(X_train, Y_train, 'rx')
    plt.legend()
    if save_path:
        plt.savefig(save_path)