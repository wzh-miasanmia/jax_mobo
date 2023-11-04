from enum import Enum
from functools import partial
from typing import Callable, Union

from jax.scipy.stats import norm

from bayex.gp import GParameters, predict
from bayex.types import Array


def expected_improvement(
    x_pred: Array,
    params: GParameters,
    noise: float,
    x: Array,
    y: Array,
    dtypes: Union[dict, None],
    xi=0.01,
) -> Array:
    y_max = y.max()
    mu, std = predict(params, noise, x, y, dtypes, xt=x_pred)
    improvement = mu.T - y_max - xi
    z = improvement / (std + 1e-3)
    ei = improvement * norm.cdf(z) + std * norm.pdf(z)
    return ei

