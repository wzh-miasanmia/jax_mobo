import jax.numpy as jnp
from jax import jit
import matplotlib.pyplot as plt
from jax.scipy.linalg import cholesky, solve_triangular
from matplotlib import animation, cm


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
    L = cholesky(K, lower=True) 
    alpha = solve_triangular(L.T, solve_triangular(L, Y_train, lower=True), lower=False)

    # Compute the posterior mean vector mu_s
    mu_s = K_s.T.dot(alpha)  

    # Calculate the posterior covariance matrix cov_s
    v = solve_triangular(L, K_s, lower=True)
    cov_s = K_ss - v.T.dot(v)  
    
    return mu_s, cov_s

posterior_jit = jit(posterior)

# 
@jit
def nll_fn(X_train, Y_train, noise, naive=False):
    """
    Returns a function that computes the negative log marginal
    likelihood for training data X_train and Y_train and given
    noise level.

    Args:
        X_train: training locations (m x d).
        Y_train: training targets (m x 1).
        noise: known noise level of Y_train.
        naive: if True use a naive implementation , if
               False use a numerically more stable implementation.

    Returns:
        Minimization objective.
    """
    Y_train = Y_train.ravel()
    
    def nll_naive(theta):
        K = kernel(X_train, X_train, l=theta[0], sigma_f=theta[1]) + \
            noise**2 * jnp.eye(len(X_train))
        return 0.5 * jnp.linalg.slogdet(K) + \
               0.5 * jnp.dot(Y_train, solve_triangular(cholesky(K), Y_train, lower=True)) + \
               0.5 * len(X_train) * jnp.log(2*jnp.pi)
        
    def nll_stable(theta):
        K = kernel(X_train, X_train, l=theta[0], sigma_f=theta[1]) + \
            noise**2 * jnp.eye(len(X_train))
        L = cholesky(K)
        S1 = solve_triangular(L, Y_train, lower=True)
        S2 = solve_triangular(L.T, S1, lower=False)
        return jnp.sum(jnp.log(jnp.diag(L))) + \
               0.5 * jnp.dot(Y_train, S2) + \
               0.5 * len(X_train) * jnp.log(2*jnp.pi)

    if naive:
        return nll_naive
    else:
        return nll_stable

nll_jit = jit(nll_fn)

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

def plot_gp_2D(gx, gy, mu, X_train, Y_train, title, i):
    ax = plt.gcf().add_subplot(1, 2, i, projection='3d')
    ax.plot_surface(gx, gy, mu.reshape(gx.shape), cmap=cm.coolwarm, linewidth=0, alpha=0.2, antialiased=False)
    ax.scatter(X_train[:,0], X_train[:,1], Y_train, c=Y_train, cmap=cm.coolwarm)
    ax.set_title(title)