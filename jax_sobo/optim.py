import jax
from gp import posterior_jit
from jax.scipy.stats import norm
import jax.numpy as jnp
from jax import random

from scipy.optimize import minimize
import matplotlib.pyplot as plt

# The acq used in the article is EHVI, EI is used here instead.


def expected_improvement(X:jax.Array, X_sample:jax.Array, Y_sample:jax.Array, ls=1.0, sigma_f=1.0, noise=1e-8, xi=0.01):
    '''
    Computes the EI at points X based on existing samples X_sample
    and Y_sample using a Gaussian process surrogate model.
    
    Args:
        X: Points at which EI shall be computed (m x d).
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        xi: Exploitation-exploration trade-off parameter.
    
    Returns:
        Expected improvements at points X.
    '''
    mu, sigma = posterior_jit(X_s=X, X_train=X_sample, Y_train=Y_sample,l=ls, sigma_f=sigma_f, sigma_y=noise)
    mu_sample, _ = posterior_jit(X_s=X_sample, X_train=X_sample, Y_train=Y_sample,l=ls, sigma_f=sigma_f, sigma_y=noise)
    mu_sample_opt = jnp.max(mu_sample)
    improvement = mu.T - mu_sample_opt - xi
    z = improvement / (sigma + 1e-3)
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    return ei

def propose_location(acquisition, X, X_sample, Y_sample, bounds, ls=1.0, sigma_f=1.0, noise=1e-8, n_restarts=25):
    '''
    Proposes the next sampling point by optimizing the acquisition function.
    
    Args:
        acquisition: Acquisition function.
        X: Points at which EI shall be computed (m x d).
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        gpr: A GaussianProcessRegressor fitted to samples.

    Returns:
        Location of the acquisition function maximum.
    '''
    dim = X_sample.shape[1]
    min_val = 1
    min_x = None
    key = random.PRNGKey(42)
    
    def min_obj(X):
        # Minimization objective is the negative acquisition function
        return -acquisition(X.reshape(-1, dim), X_sample, Y_sample, ls=ls, sigma_f=ls, noise=noise).flatten()
    

    
    # Find the best optimum by starting from n_restart different random points.
    for x0 in random.uniform(key, (n_restarts, dim), minval=bounds[:, 0], maxval=bounds[:, 1]):
        res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')        
        if res.fun < min_val:
            min_val = res.fun[0]
            min_x = res.x           
            
    return min_x.reshape(-1, 1)

def plot_approximation(mu, std, X, Y, X_sample, Y_sample, X_next=None, show_legend=False):
    plt.fill_between(X.ravel(), 
                    mu.ravel() + 1.96 * std, 
                    mu.ravel() - 1.96 * std, 
                    alpha=0.1) 
    plt.plot(X, Y, 'y--', lw=1, label='Noise-free objective')
    plt.plot(X, mu, 'b-', lw=1, label='Surrogate function')
    plt.plot(X_sample, Y_sample, 'kx', mew=3, label='Noisy samples')
    if X_next:
        plt.axvline(x=X_next, ls='--', c='k', lw=1)
    if show_legend:
        plt.legend()
        
def plot_acquisition(X, Y, X_next, show_legend=False):
    plt.plot(X, Y, 'r-', lw=1, label='Acquisition function')
    plt.axvline(x=X_next, ls='--', c='k', lw=1, label='Next sampling location')
    if show_legend:
        plt.legend() 