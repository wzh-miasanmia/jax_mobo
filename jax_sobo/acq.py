from gp_reference import GParameters, predict
from jax.scipy.stats import norm

# The acq used in the article is EHVI, EI is used here instead.


def expected_improvement(params: GParameters, x, y, x_pred, xi=0.01):
    y_max = y.max() # the best sample so far
    mu, std = predict(params, x, y, x_new=x_pred)
    if std == 0:
        return 0
    improvement = mu.T - y_max - xi
    z = improvement / std
    ei = improvement * norm.cdf(z) + std * norm.pdf(z)

    return ei

import jax.numpy as jnp
from scipy.stats import norm

def expected_improvement(X, X_sample, Y_sample, gpr, xi=0.01):
    """
    Computes the EI at points X based on existing samples X_sample
    and Y_sample using a Gaussian process surrogate model.
    
    Args:
        X: Points at which EI shall be computed (m x d).
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        gpr: A GaussianProcessRegressor fitted to samples.
        xi: Exploitation-exploration trade-off parameter.
    
    Returns:
        Expected improvements at points X.
    """
    mu, sigma = gpr.predict(X)
    mu_sample = gpr.predict(X_sample)

    sigma = sigma.reshape((-1, 1))
    
    # Needed for noise-based model,
    # otherwise use jnp.max(Y_sample).
    # See also section 2.4 in [1]
    mu_sample_opt = jnp.max(mu_sample)

    with jnp.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei = jnp.where(sigma == 0.0, 0.0, ei)

    return ei
