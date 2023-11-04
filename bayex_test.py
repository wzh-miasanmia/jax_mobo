import bayesian_jax

noise = 0.01
def f(x, y):
    return -y ** 2 - (x - y) ** 2 + 3 * x / y - 2 + noise

constrains = {'x': (-10, 10), 'y': (0, 10)}
optim_params = bayesian_jax.optim(f, noise, constrains=constrains, seed=42, n=10)

show_results(optim_params)


# import bayex
# import jax.numpy as jnp

# def f(X):
#     return -jnp.sin(3 * X) - X**2 + 0.7 * X

# constrains = {'X': (-1.0, 2.0)}
# optim_params = bayex.optim(f, constrains=constrains, seed=42, n=20)

# bayex.show_results(optim_params)
# print(optim_params.target, optim_params.params)

