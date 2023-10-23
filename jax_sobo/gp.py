import jax.numpy as jnp
from jax import jit
import matplotlib.pyplot as plt

# Covariance matrix calculation
def kernel(X1, X2, l=1.0, sigma_f=1.0):
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


def posterior(X_s, X_train, Y_train, l=1.0, sigma_f=1.0, sigma_y=1e-8):
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
    L = jnp.linalg.cholesky(K)
    alpha = jnp.linalg.solve(L.T, jnp.linalg.solve(L, Y_train))

    # Compute the posterior mean vector mu_s
    mu_s = K_s.T.dot(alpha)  

    # Calculate the posterior covariance matrix cov_s
    v = jnp.linalg.solve(L, K_s)
    cov_s = K_ss - v.T.dot(v)  
    
    return mu_s, cov_s

posterior_jit = jit(posterior)


# Plots the results
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