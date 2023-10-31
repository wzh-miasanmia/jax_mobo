import jax
from gp import predict
from jax.scipy.stats import norm
import jax.numpy as jnp
from jax import jacrev, jit, lax, random, tree_map, vmap, Array
from typing import Callable, NamedTuple, Tuple, Union, Optional

from scipy.optimize import minimize
import matplotlib.pyplot as plt
from functools import partial

# The acq used in the article is EHVI, EI is used here instead.
class OptimizerParameters(NamedTuple):
    """
    Object holding the results of the optimization.
    """

    target: Union[Array, float]
    params: Array
    f: Callable
    params_all: Array
    target_all: Array

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
    mu, sigma = predict(X_s=X, X_train=X_sample, Y_train=Y_sample,l=ls, sigma_f=sigma_f, sigma_y=noise, return_std=True, return_cov=False)
    mu_sample = predict(X_s=X_sample, X_train=X_sample, Y_train=Y_sample,l=ls, sigma_f=sigma_f, sigma_y=noise, return_std=False, return_cov=False)
    mu_sample_opt = jnp.max(mu_sample)
    improvement = mu.T - mu_sample_opt - xi
    z = improvement / (sigma + 1e-3)
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    return ei

def jacobian(f: Callable) -> Callable:
    return jit(jacrev(f))


@partial(jit, static_argnums=(6,7,8))
def suggest_next(
    key, 
    X_sample, 
    Y_sample, 
    bounds, 
    ls=1.0, 
    sigma_f=1.0, 
    noise=1e-8, 
    n_seed=1000, 
    lr=0.1, 
    n_epochs=150):

    key1, key2 = random.split(key, 2)
    dim = bounds.shape[0]

    domain = random.uniform(
        key1, shape=(n_seed, dim), minval=bounds[:, 0], maxval=bounds[:, 1]
    )

    _acq = partial(expected_improvement, X_sample=X_sample, Y_sample=Y_sample, ls=ls, sigma_f=sigma_f, noise=noise, xi=0.01)

    J = jacobian(lambda x: _acq(x.reshape(-1, dim)).reshape())
    HS = vmap(lambda x: x + lr * J(x))
    
    domain = lax.fori_loop(0, n_epochs, lambda _, d: HS(d), domain)
    domain = jnp.clip(
        domain.reshape(-1, dim), a_min=bounds[:, 0], a_max=bounds[:, 1]
    )

    ys = _acq(domain)
    next_X = domain[ys.argmax()]
    return next_X, key2


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