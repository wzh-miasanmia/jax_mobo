# import bayesian_jax

# def f(x, y):
#     return -y ** 2 - (x - y) ** 2 + 3 * x / y - 2

# constrains = {'x': (-10, 10), 'y': (0, 10)}
# optim_params = bayesian_jax.optim(f, constrains, noise, seed=42, n=10)

# bayesian_jax.show_results(optim_params)


import bayesian_jax
import jax.numpy as jnp

def f(X):
    return -jnp.sin(3 * X) - X**2 + 0.7 * X

constrains = {'X': (-1.0, 2.0)}
optim_params = bayesian_jax.optim(f, constrains=constrains, seed=42, n=20, plot_figure=True, path='test.png')

bayesian_jax.show_results(optim_params)
print(optim_params.target, optim_params.params)

