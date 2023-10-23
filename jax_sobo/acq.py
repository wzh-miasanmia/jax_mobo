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
