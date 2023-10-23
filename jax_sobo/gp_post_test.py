import gp
import jax.numpy as jnp
from jax import jacrev, jit, lax, random, tree_map, vmap
import matplotlib.pyplot as plt
from jax import random
from functools import partial
import numpy as np

key = random.PRNGKey(42)

X = jnp.arange(-5, 5, 0.2).reshape(-1, 1)

X_train = jnp.array([-4, -3, -2, -1, 1]).reshape(-1, 1)

dim = 1

def f(x):
    return jnp.sin(x)
 
# no noise y
Y_train = vmap(f)(*X_train.T)

mu_s , cov_s = gp.posterior_jit(X, X_train, Y_train)

import numpy as np

samples = np.random.multivariate_normal(mu_s.ravel(), cov_s, 3)
# samples = random.multivariate_normal(key, mu_s.ravel(), cov_s, (3,))
save_path='/home/wzhmiasanmia/ma_workspace/jax_mobo/jax_sobo/gp_post_1.png'
gp.plot_gp(mu_s, cov_s, X, X_train, Y_train, samples=samples, save_path=save_path)


noise = 0.4

# Noisy training data
X_train = jnp.arange(-3, 4, 1).reshape(-1, 1)
noise_values = noise * random.normal(key, shape=X_train.shape)
Y_train = jnp.sin(X_train) + noise_values
# Compute mean and covariance of the posterior distribution
mu_s, cov_s = gp.posterior_jit(X, X_train, Y_train, sigma_y=noise)

samples = np.random.multivariate_normal(mu_s.ravel(), cov_s, 3)
save_path='/home/wzhmiasanmia/ma_workspace/jax_mobo/jax_sobo/gp_post_2.png'
gp.plot_gp(mu_s, cov_s, X, X_train=X_train, Y_train=Y_train, samples=samples,save_path=save_path)