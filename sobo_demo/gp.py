import jax
import jax.numpy as jnp
from collections import namedtuple
from jax import grad, jit, tree_util, vmap, lax
import matplotlib.pyplot as plt
from jax.scipy.linalg import cholesky, solve_triangular
from matplotlib import animation, cm
from jax.lax import fori_loop
from typing import Tuple
from functools import partial

GP_parameters = namedtuple("GP_parameters", [ "lengthscale", "amplitude"])

# Covariance matrix calculation
def kernel(X1:jax.Array, X2:jax.Array, l=1.0, sigma_f=1.0):
    """
    Isotropic squared exponential kernel.
    
    Args:
        X1: Array of m points (m x d).
        X2: Array of n points (n x d).

    Returns:
        (m x n) matrix.
    """
    sqdist = jnp.sum(X1**2, axis=1).reshape((-1, 1)) + jnp.sum(X2**2, axis=1) - 2 * jnp.dot(X1, X2.T)
    return sigma_f**2 * jnp.exp(-0.5 / l**2 * sqdist)

kernel_jit = jit(kernel)


@partial(jit, static_argnums=(6, 7))
def predict(X_s:jax.Array, X_train:jax.Array, Y_train:jax.Array, l=1.0, sigma_f=1.0, sigma_y=1e-8,return_std=False, return_cov=True):
    """
    Computes the suffifient statistics of the posterior distribution 
    from m training data X_train and Y_train and n new inputs X_s.
    
    Args:
        X_s: New input locations (n x d).
        X_train: Training locations (m x d).
        Y_train: Training targets (m x 1).
        l: Kernel length parameter.
        sigma_f: Kernel vertical variation parameter.
        sigma_y: Noise parameter.
    
    Returns:
        Posterior mean vector (n x d) and covariance matrix (n x n).
    """
    K = kernel_jit(X_train, X_train, l, sigma_f) + sigma_y**2 * jnp.eye(len(X_train)) 
    K_s = kernel_jit(X_train, X_s, l, sigma_f)  
    K_ss = kernel_jit(X_s, X_s, l, sigma_f) + 1e-8 * jnp.eye(len(X_s))  

    # Use Cholesky decomposition instead of inverse
    L = cholesky(K, lower=True) 
    alpha = solve_triangular(L.T, solve_triangular(L, Y_train, lower=True), lower=False)

    # Compute the posterior mean vector mu_s
    mu_s = K_s.T.dot(alpha)  

    # Calculate the posterior covariance matrix cov_s
    v = solve_triangular(L, K_s, lower=True)
    cov_s = K_ss - v.T.dot(v)  
    
    if return_std and return_cov:
        return mu_s, jnp.sqrt(jnp.diag(cov_s)), cov_s
    elif return_std:
        return mu_s, jnp.sqrt(jnp.diag(cov_s))
    elif return_cov:
        return mu_s, cov_s
    else:
        return mu_s


def softplus(x):
    return jnp.logaddexp(x, 0.0)

def mll(params, X_train, Y_train, noise):
    ls,amp = tree_util.tree_map(softplus, params)
    K = kernel_jit(X_train, X_train, l=ls, sigma_f=amp) + \
        noise**2 * jnp.eye(len(X_train))
    L = cholesky(K)
    S1 = solve_triangular(L, Y_train, lower=True)
    S2 = solve_triangular(L.T, S1, lower=False)
    mll =  jnp.sum(jnp.log(jnp.diag(L))) + \
            0.5 * jnp.dot(Y_train, S2) + \
            0.5 * len(X_train) * jnp.log(2*jnp.pi)
    return mll

grad_fun = jit(grad(mll))

def optimize_mll(
    params,
    X_train:jax.Array, 
    Y_train:jax.Array, 
    momentums,
    scales,
    noise, 
    num_steps: int = 20, 
    lr: float = 0.01,
):

    Y_train = Y_train.ravel()

    def train_step(
        params, momentums, scales,
    ):
        grads = grad_fun(params, X_train, Y_train, noise)
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
        num_steps,
        lambda _, v: train_step(*v),
        (params, momentums, scales),
    )

    return params, momentums, scales





## Plots the results ##
def plot_gp(mu, cov, X, X_train=None, Y_train=None, samples=[], save_path=None):
    """
    Plots the results of a Gaussian Process regression.

    Args:
        mu: Array representing the mean values of the Gaussian Process predictions.
        cov: Covariance matrix of the Gaussian Process predictions.
        X: Input locations for plotting.
        X_train: Training input locations (if available).
        Y_train: Training target values (if available).
        samples: List of sample predictions to be plotted.
        save_path: Path to save the plot as an image file (e.g., 'plot.png').

    Returns:
        None
    """
    X = X.ravel()
    mu = mu.ravel()
    uncertainty = 1.96 * jnp.sqrt(jnp.diag(cov))
    plt.figure()
    plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.1)
    plt.plot(X, mu, label='Mean')
    for i, sample in enumerate(samples):
        plt.plot(X, sample, lw=1, ls='--', label=f'Sample {i+1}')
    if X_train is not None:
        plt.plot(X_train, Y_train, 'rx')
    plt.legend()
    plt.show()
    if save_path:
        plt.savefig(save_path)

def plot_gp_2D(gx, gy, mu, X_train, Y_train, title, i):
    ax = plt.gcf().add_subplot(1, 2, i, projection='3d')
    ax.plot_surface(gx, gy, mu.reshape(gx.shape), cmap=cm.coolwarm, linewidth=0, alpha=0.2, antialiased=False)
    ax.scatter(X_train[:,0], X_train[:,1], Y_train, c=Y_train, cmap=cm.coolwarm)
    ax.set_title(title)