import jax
import jax.numpy as jnp
from jax import grad, jit, tree_util, vmap
import matplotlib.pyplot as plt
from jax.scipy.linalg import cholesky, solve_triangular
from matplotlib import animation, cm
from jax.lax import fori_loop
from typing import Tuple



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


def posterior(X_s:jax.Array, X_train:jax.Array, Y_train:jax.Array, l=1.0, sigma_f=1.0, sigma_y=1e-8,return_std=False, return_cov=False):
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

posterior_jit = jit(posterior)


def optimize_mll(initial_theta:jax.Array, X_train:jax.Array, Y_train:jax.Array, noise, num_steps, lr, method:str):
    """
    Optimize the negative log marginal likelihood for Gaussian Process Regression.

    Parameters:
    -----------
    initial_theta : jax.Array
        Initial values for the hyperparameters to be optimized.
    X_train : jax.Array
        Training locations (m x d).
    Y_train : jax.Array
        Training targets (m x 1).
    noise : float
        Known noise level of the training targets.
    num_steps : int
        Number of optimization steps to perform.
    lr : float
        Learning rate for the optimization.

    Returns:
    --------
    jax.Array
        Optimized hyperparameters that minimize the negative log marginal likelihood.
    """
    Y_train = Y_train.ravel()

    def mll(theta):
        K = kernel_jit(X_train, X_train, l=theta[0], sigma_f=theta[1]) + \
            noise**2 * jnp.eye(len(X_train))
        L = cholesky(K)
        S1 = solve_triangular(L, Y_train, lower=True)
        S2 = solve_triangular(L.T, S1, lower=False)
        return jnp.sum(jnp.log(jnp.diag(L))) + \
                0.5 * jnp.dot(Y_train, S2) + \
                0.5 * len(X_train) * jnp.log(2*jnp.pi)
    grad_fun = jit(grad(mll))

    theta = initial_theta
    momentums = jnp.zeros_like(theta)
    scales = jnp.zeros_like(theta)
    if method == 'SGD':
        for _ in range(num_steps):
            grads = grad_fun(theta)

            momentums = 0.9 * momentums + 0.1 * grads

            scales = 0.9 * scales + 0.1 * grads ** 2

            theta -= lr * momentums / jnp.sqrt(scales + 1e-5)

        return theta
    
    elif method == 'BFGS':
        return 'BFGS is not finish yet'
    else:
        return 'Please choose a method correctly'




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