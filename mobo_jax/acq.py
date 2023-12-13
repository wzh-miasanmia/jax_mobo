from enum import Enum
from functools import partial
from typing import Callable, Union, List
import numpy as np
from jax.scipy.stats import norm
import jax.numpy as jnp
from mobo_jax.gp import GParameters, predict
from mobo_jax.types import Array
from mobo_jax.pareto_cal import is_non_dominated_np
from mobo_jax.HV_cal import Hypervolume
import jax
class ACQ(Enum):
    EI = 1
    POI = 2
    UCB = 3
    LCB = 4
    EHVI = 5

def select_acq(acq: Union[ACQ, str], acq_params: dict) -> Callable:
    """
    Wrapper that selects the correct acquisition function and makes sure
    that the parameters for it exist.
    """
    # note(alonfnt): 3.10 asks for a switch statement here, but for <3.10
    # if/else makes more sense.

    if acq == ACQ.EI:
        xi = acq_params["xi"] if "xi" in acq_params else 0.01
        return partial(expected_improvement, xi=xi)
    elif acq == ACQ.POI:
        xi = acq_params["xi"] if "xi" in acq_params else 0.01
        return partial(probability_improvement, xi=xi)
    elif acq == ACQ.UCB:
        kappa = acq_params["kappa"] if "kappa" in acq_params else 0.01
        return partial(upper_confidence_bounds, kappa=kappa)
    elif acq == ACQ.LCB:
        kappa = acq_params["kappa"] if "kappa" in acq_params else 0.01
        return partial(lower_confidence_bounds, kappa=kappa)
    elif acq == ACQ.EHVI:
        return expected_HV_improvement
    raise ValueError("The acquisition function given is not correct.")


def expected_improvement(
    x_pred: Array,
    params: GParameters,
    x: Array,
    y: Array,
    dtypes: Union[dict, None],
    xi: dict,
) -> Array:
    y_max = y.max()
    mu, std = predict(params, x, y, dtypes, xt=x_pred)
    improvement = mu.T - y_max - xi
    z = improvement / (std + 1e-3)
    ei = improvement * norm.cdf(z) + std * norm.pdf(z)
    return ei


def probability_improvement(
    x_pred: Array,
    params: GParameters,
    x: Array,
    y: Array,
    dtypes: Union[dict, None],
    xi: float,
) -> Array:
    y_max = y.max()
    mu, std = predict(params, x, y, dtypes, xt=x_pred)
    z = (mu - y_max - xi) / std
    return norm.cdf(z)


def upper_confidence_bounds(
    x_pred: Array,
    params: GParameters,
    x: Array,
    y: Array,
    dtypes: Union[dict, None],
    kappa: float,
) -> Array:
    mu, std = predict(params, x, y, dtypes, xt=x_pred)
    return mu + kappa * std


def lower_confidence_bounds(
    x_pred: Array,
    params: GParameters,
    x: Array,
    y: Array,
    dtypes: Union[dict, None],
    kappa: float,
) -> Array:
    mu, std = predict(params, x, y, dtypes, xt=x_pred)
    return mu - kappa * std


def expected_HV_improvement(
    x_pred: Array,
    params_list: List,
    x: Array,
    y_multi: Array,
    dtypes: Union[dict, None],
    ref_point: Array,
) -> Array:
    # definition of hypervolume
    HV = Hypervolume(ref_point)
    
    # previous pareto and hypervolume
    mu_sample_list, std_list = zip(*[predict(params, x, y, dtypes, xt=x) for y, params in zip(y_multi.T, params_list)])
    previous_points = jnp.column_stack(mu_sample_list)
    previous_pareto_mask = is_non_dominated_np(previous_points)
    previous_pareto = previous_points[previous_pareto_mask]
    HV_previous = HV.compute(previous_pareto)
    
    # new pareto and hypervolume
    mu_list, std_list = zip(*[predict(params, x, y, dtypes, xt=x_pred) for y, params in zip(y_multi.T, params_list)])
    new_point = jnp.column_stack(mu_list)
    new_points = jnp.vstack((new_point, previous_points))
    new_pareto_mask = is_non_dominated_np(new_points)
    new_pareto = new_points[new_pareto_mask]
    HV_new = HV.compute(new_pareto)
    
    # calculate ehvi
    improvement = HV_new - HV_previous
    sigma_aggregated = jnp.linalg.norm(jnp.column_stack(std_list))
    Z = improvement / sigma_aggregated
    ehvi = improvement * norm.cdf(Z) + sigma_aggregated * norm.pdf(Z)
    # ehvi[sigma_aggregated == 0.0] = 0.0
    return ehvi