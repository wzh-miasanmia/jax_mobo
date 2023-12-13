import jax.numpy as jnp

def schaffer_function_n1_multiobjective(x):
    """
    Multi-objective Schaffer Function N. 1

    Parameters:
    x (numpy array): Input variable x = [x1]

    Returns:
    numpy array: The results of the multi-objective Schaffer Function N. 1
    """
    f1 = x**2
    f2 = (x - 2)**2
    return jnp.array([-f1, -f2]) # maximum


def poloni_objectives(x):
    A1 = 0.5 * jnp.sin(1) - 2 * jnp.cos(1) + jnp.sin(2) - 1.5 * jnp.cos(2)
    A2 = 1.5 * jnp.sin(1) - jnp.cos(1) + 2 * jnp.sin(2) - 0.5 * jnp.cos(2)

    B1 = 0.5 * jnp.sin(x[0]) - 2 * jnp.cos(x[0]) + jnp.sin(x[1]) - 1.5 * jnp.cos(x[1])
    B2 = 1.5 * jnp.sin(x[0]) - jnp.cos(x[0]) + 2 * jnp.sin(x[1]) - 0.5 * jnp.cos(x[1])

    f1 = 1 + (A1 - B1)**2 + (A2 - B2)**2
    f2 = (x[0] + 3)**2 + (x[1] + 1)**2

    return jnp.array([-f1, -f2]) # maximum


def viennet_function(x):
    f1 = 0.5 * (x[0]**2 + x[1]**2) + jnp.sin(x[0]**2 + x[1]**2)
    f2 = (3*x[0]-2*x[1]+4)**2 / 8 + (x[0]-x[1]+1)**2 / 27 + 15
    f3 = 1 / (x[0]**2+x[1]**2+1) - 1.1 * jnp.exp(-(x[0]**2+x[1]**2))
    
    return jnp.array([-f1, -f2, -f3]) # maximum