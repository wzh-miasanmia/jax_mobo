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
    return jnp.array([f1, f2])

def kursawe_function(x):
    """
    Kursawe Function

    Parameters:
    x (numpy array): Input variables x = [x1, x2]

    Returns:
    numpy array: The results of the Kursawe Function
    """
    f1 = -10 * jnp.exp(-0.2 * jnp.sqrt(x[0]**2 + x[1]**2))
    f2 = (jnp.abs(x[0])**0.8 + 5 * jnp.sin(x[0]**3) + 3 * jnp.cos(5 * x[0])) +\
         (jnp.abs(x[1])**0.8 + 5 * jnp.sin(x[1]**3) + 3 * jnp.cos(5 * x[1]))
    return np.array([f1, f2])