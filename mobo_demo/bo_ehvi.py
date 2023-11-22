from gp import predict
import jax.numpy as jnp
from jax.scipy.stats import norm


def pareto_frontier(Y):
    return jnp.any(Y[:, None] <= Y, axis=-1)


def hypervolume(Y): # need to change
    return jnp.prod(jnp.max(Y, axis=0) - jnp.min(Y, axis=0))

def ehvi(
    X:jax.Array, 
    X_sample:jax.Array, 
    Y_sample:jax.Array, 
    ls=1.0, 
    sigma_f=1.0, 
    noise=1e-8,):# need to change

    mu, sigma = predict(X_s=X, X_train=X_sample, Y_train=Y_sample,\
        l=ls, sigma_f=sigma_f, sigma_y=noise, return_std=True, return_cov=False)

    improvement = current_hypervolume - hypervolume(np.vstack([Y_current_pareto, mu]))
    Z = (current_pareto_frontier * improvement).sum(axis=-1)

    return -Z


@partial(jit, static_argnums=(6,7,8)) # need to change
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
    # need to change
    _acq = partial(ehvi, X_sample=X_sample, Y_sample=Y_sample, ls=ls, sigma_f=sigma_f, noise=noise, xi=0.01)

    J = jacobian(lambda x: _acq(x.reshape(-1, dim)).reshape())
    HS = vmap(lambda x: x + lr * J(x))
    
    domain = lax.fori_loop(0, n_epochs, lambda _, d: HS(d), domain)
    domain = jnp.clip(
        domain.reshape(-1, dim), a_min=bounds[:, 0], a_max=bounds[:, 1]
    )
    domain, key = replace_nan_values(domain, bounds[:, 0], bounds[:, 1], key2)
    ys = _acq(domain)
    next_X = domain[ys.argmax()]
    return next_X, key