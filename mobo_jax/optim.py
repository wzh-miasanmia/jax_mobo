from functools import partial
from typing import Callable, NamedTuple, Tuple, Union, Optional, List

import jax.numpy as jnp
from jax import jacrev, jit, lax, random, tree_map, vmap

from mobo_jax.acq import ACQ, select_acq
from mobo_jax.gp import DataTypes, GParameters, round_integers, train, predict
from mobo_jax.types import Array
from mobo_jax.pareto_cal import is_non_dominated_np
from mobo_jax.HV_cal import Hypervolume
import matplotlib.pyplot as plt

def jacobian(f: Callable) -> Callable:
    return jit(jacrev(f))


def replace_nan_values(arr: Array) -> Array:
    """
    Replaces the NaN values (if any) in arr with 0.

    Parameters:
    -----------
    arr: The array where NaN is removed from.

    Returns:
    --------
    The array with all the NaN elements replaced with 0.
    """
    # todo(alonfnt): Find a more robust solution.
    return jnp.where(jnp.isnan(arr), 0, arr)


@partial(jit, static_argnums=(6,))
def suggest_next(
    key: Array,
    params_list: List,
    x: Array,
    y: Array,
    bounds: Array,
    dtypes: DataTypes,
    acq: Callable,
    ref_point: Array,
    n_seed: int = 1000,
    lr: float = 0.1,
    n_epochs: int = 150,
) -> Tuple[Array, Array]:
    """
    Suggests the new point to sample by optimizing the acquisition function.

    Parameters:
    -----------
    key: The pseudo-random generator key used for jax random functions.
    params: Hyperparameters of the Gaussian Process Regressor.
    x: Sampled points.
    y: Sampled targets.
    bounds: Array of (2, dim) shape with the lower and upper bounds of the
            variables.y_max: The current maximum value of the target values Y.
    dtypes: The type of non-real variables in the target function.
    n_seed (optional): the number of points to probe and minimize until
            finding the one that maximizes the acquisition functions.
    lr (optional): The step size of the gradient descent.
    n_epochs (optional): The number of steps done on the descent to minimize
            the seeds.


    Returns:
    --------
    A tuple with the parameters that maximize the acquisition function and a
    jax PRGKey to be used in the next sampling.
    """

    key1, key2 = random.split(key, 2)
    dim = bounds.shape[0]

    domain = random.uniform(
        key1, shape=(n_seed, dim), minval=bounds[:, 0], maxval=bounds[:, 1]
    )

    _acq = partial(acq, params_list=params_list, x=x, y_multi=y, dtypes=dtypes, ref_point=ref_point)

    J = jacobian(lambda x: _acq(x.reshape(-1, dim)).reshape())
    HS = vmap(lambda x: x + lr * J(x))

    domain = lax.fori_loop(0, n_epochs, lambda _, d: HS(d), domain)
    domain = jnp.clip(
        domain.reshape(-1, dim), a_min=bounds[:, 0], a_max=bounds[:, 1]
    )
    domain = replace_nan_values(domain)
    domain = round_integers(domain, dtypes)

    ys = _acq(domain)
    next_X = domain[ys.argmax()]
    return next_X, key2


@partial(jit, static_argnums=(1, 2))
def _extend_array(arr: Array, pad_width: int, axis: int) -> Array:
    """
    Extends the array pad_width only on one direction and fills it with
    the last value of that axis.
    TODO: consider donate_argnums=0 if the device allows it.
    """
    pad_shape = [(0, 0)] * arr.ndim
    pad_shape[axis] = (0, pad_width)
    return jnp.pad(arr, pad_shape, mode="edge")

def from_X_calculate_HV(X, Y_multi, X_sample, params_list, dtypes, ref_point):
    mu_sample_list = []
    for _, (Y, params) in enumerate(zip(Y_multi, params_list)):
        mu_sample, std = predict(params, X, Y, dtypes, xt=X_sample)
        mu_sample_list.append(mu_sample)
    mu_samples = (jnp.vstack(mu_sample_list)).T # (n_samples, n_functions)
    pareto_mask = is_non_dominated_np(mu_samples)
    pareto = mu_samples[pareto_mask]    
    HV_class = Hypervolume(ref_point)
    HV = HV_class.compute(pareto)
    return HV

def optim(
    f_multi: Callable,
    constraints: dict,
    seed: int = 42,
    n_init: int = 5,
    n: int = 10,
    ctypes: Optional[dict] = None,
    acq: ACQ = ACQ.EHVI,
    **acq_params: dict,
) -> Array:
    """
    Finds the inputs of 'f' that yield the maximum value between the given
    'constrains', after 'n_init' + 'n' iterations.

    Parameters:
    -----------
    f: Function to optimize.
    constrains: Dictionary with the domain of each input variable.
    seed: Pseudo-random number generator seed for reproducibility
    n_init: Number of initial evaluations before suggesting optimized samples.
    n: Number of sampling iterations.
    ctypes: The type of non-real variables in the target function.

    Returns:
    --------
    The parameters that maximize the given 'f'.
    """

    assert n > 0, "Num of iterations n should be a positive integer"

    key = random.PRNGKey(seed)
    dim = len(constraints)
    _vars = f_multi.__code__.co_varnames[: f_multi.__code__.co_argcount]
    _sorted_constrains = {k: constraints[k] for k in _vars}

    if ctypes is not None:
        _sorted_types = {k: ctypes[k] for k in _vars if k in ctypes}
        dtypes = DataTypes(
            integers=[
                _vars.index(k) for k, v in _sorted_types.items() if v == int
            ]
        )
    else:
        dtypes = DataTypes(integers=[])

    _acq = select_acq(acq, acq_params)

    bounds = jnp.asarray(list(_sorted_constrains.values()))

    X = random.uniform(
        key,
        shape=(n_init, dim),
        minval=bounds[:, 0],
        maxval=bounds[:, 1],
    )
    X = round_integers(X, dtypes)
    Y_multi = vmap(f_multi)(*X.T)

    # define ref_point according to Appendix A.2
    min_values = jnp.min(Y_multi, axis=0)
    max_values = jnp.max(Y_multi, axis=0)
    ref_point = ((min_values + max_values) / 2).squeeze()
    
    # Expand the array with the same last values to not perjudicate the gp.
    # the reason to apply it as a function is to avoid having twice the memory
    # usage, since JAX does not do inplace updates except after being
    # compiled.
    X = _extend_array(X, n, 0)
    Y_multi = _extend_array(Y_multi, n, 0)
    n_f = Y_multi.shape[1]

    # Initialize the GP parameters
    params_list = []
    momentums_list = []
    scales_list = []
    for _ in range(n_f):
        params = GParameters(
            noise=jnp.zeros((1, 1)) - 5.0,
            amplitude=jnp.zeros((1, 1)),
            lengthscale=jnp.zeros((1, dim)),
        )
        momentums = tree_map(lambda x: x * 0, params)
        scales = tree_map(lambda x: x * 0 + 1, params)
        
        params_list.append(params)
        momentums_list.append(momentums)
        scales_list.append(scales)


    for idx in range(n_init, n + n_init):
        for d in range(n_f):
            # optim the hyperparameters of GP
            params, momentums, scales = train(
                x=X, y=Y_multi[:,d], params=params_list[d], momentums=momentums_list[d], scales=scales_list[d], dtypes=dtypes) 
            params_list[d], momentums_list[d], scales_list[d] = params, momentums, scales


        max_params, key = suggest_next(key, params_list, X, Y_multi, bounds, dtypes, _acq, ref_point)
        X = X.at[idx, ...].set(max_params)
        Y_multi = Y_multi.at[idx].set(f_multi(*max_params))

    # update the best hyperparameters of GP
    for d in range(n_f):
        params, momentums, scales = train(
            X=X, Y=Y_multi[:,d], params=params_list[d], momentums=momentums_list[d], scales=scales_list[d], dtypes=dtypes) 
        params_list[d], momentums_list[d], scales_list[d] = params, momentums, scales

        
    mu_sample_list, _ = zip(*[predict(params=params, x=X, y=Y, dtypes=dtypes, xt=X) for Y, params in enumerate(zip(Y_multi, params_list))])
    mu_sample = (jnp.vstack(mu_sample_list)).T
    pareto_mask = is_non_dominated_np(mu_sample) 
    pareto = mu_sample[pareto_mask]
    return pareto
