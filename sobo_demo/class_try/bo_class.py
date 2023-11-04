import jax
from GP_class import GaussianProcessRegressor
from typing import Callable, NamedTuple, Tuple, Union, Optional
from jax.scipy.stats import norm
import jax.numpy as jnp
from jax import random, jit, grad, jacrev,lax, tree_map, vmap
from functools import partial

from scipy.optimize import minimize
import matplotlib.pyplot as plt


def expected_improvement(X, X_sample, Y_sample, gpr:GaussianProcessRegressor, xi=0.01):
    '''
    Computes the EI at points X based on existing samples X_sample
    and Y_sample using a Gaussian process surrogate model.
    
    Args:
        X: Points at which EI shall be computed (m x d).
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        gpr: A GaussianProcessRegressor fitted to samples. An instance of the GaussianProcessRegressor class from scikit-learn, already fitted to the X_sample and Y_sample.
        xi: Exploitation-exploration trade-off parameter.
    
    Returns:
        Expected improvements at points X.
    '''
    mu, sigma = gpr.predict(X, return_std=True)
    mu_sample = gpr.predict(X_sample)

    sigma = sigma.reshape(-1, 1)
    
    # Needed for noise-based model,
    # otherwise use jnp.max(Y_sample).
    # See also section 2.4 in [1]
    mu_sample_opt = jnp.max(mu_sample)

    imp = mu - mu_sample_opt - xi
    Z = imp / sigma
    ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma == 0.0] = 0.0

    return ei

def jacobian(f: Callable) -> Callable:
    return jit(jacrev(f))

@jit
def suggest_next(key, X_sample, Y_sample, gpr, bounds, acq: Callable, n_seed=1000, lr=0.1, n_epochs=150):

    key1, key2 = random.split(key, 2)
    dim = bounds.shape[0]

    domain = random.uniform(
        key1, shape=(n_seed, dim), minval=bounds[:, 0], maxval=bounds[:, 1]
    )

    _acq = partial(expected_improvement, X_sample=X_sample, Y_sample=Y_sample, gpr=gpr, xi=0.01)

    J = jacobian(lambda x: _acq(x.reshape(-1, dim)).reshape())
    HS = vmap(lambda x: x + lr * J(x))
    
    domain = lax.fori_loop(0, n_epochs, lambda _, d: HS(d), domain)
    domain = jnp.clip(
        domain.reshape(-1, dim), a_min=bounds[:, 0], a_max=bounds[:, 1]
    )

    ys = _acq(domain)
    next_X = domain[ys.argmax()]
    return next_X, key2


def propose_location(acquisition, X_sample, Y_sample, gpr, bounds, n_restarts=25):
    '''
    Proposes the next sampling point by optimizing the acquisition function.
    
    Args:
        acquisition: Acquisition function.
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
        return -acquisition(X.reshape(-1, dim), X_sample, Y_sample, gpr).flatten()
    
    # Find the best optimum by starting from n_restart different random points.
    for x0 in random.uniform(key, (n_restarts, dim), minval=bounds[:, 0], maxval=bounds[:, 1]):
        res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')        
        if res.fun < min_val:
            min_val = res.fun[0]
            min_x = res.x           
            
    return min_x.reshape(-1, 1)