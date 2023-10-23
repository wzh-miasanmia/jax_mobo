import gp
import jax.numpy as jnp
from jax import jacrev, jit, lax, random, tree_map, vmap
import matplotlib.pyplot as plt
from jax import random
from functools import partial

import numpy as np

key = random.PRNGKey(42)

X = jnp.arange(-5, 5, 0.2).reshape(-1, 1)
mu = jnp.zeros(X.shape)
cov = gp.kernel(X,X)

samples = np.random.multivariate_normal(mu.ravel(), cov, 3)
save_path='/home/wzhmiasanmia/ma_workspace/jaxfluids/jax_mobo/jax_sobo/gp_prior.png'
gp.plot_gp(mu, cov, X, samples=samples, save_path=save_path)