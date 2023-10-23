import jax.numpy as jnp
from jax import jacrev, jit, lax, random, tree_map, vmap

def suggest_next(
        key, 
        params, 
        x, 
        y, 
        bounds, 
        acq, 
        n_seed: int = 1000,
        lr: float = 0.1,
        n_epochs: int = 150 ):
    
    key1, key2 = random.split(key, 2)
    dim = bounds.shape[0]
    domain = random.uniform(
        key1, shape=(n_seed, dim), minval=bounds[:, 0], maxval=bounds[:, 1]
    )