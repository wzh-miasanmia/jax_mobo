import numpy as np

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
    return np.array([-f1, -f2]) # maximum

def kursawe_function(x):
    """
    Kursawe Function

    Parameters:
    x (numpy array): Input variables x = [x1, x2]

    Returns:
    numpy array: The results of the Kursawe Function
    """
    f1 = -10 * np.exp(-0.2 * np.sqrt(x[0]**2 + x[1]**2))
    f2 = (np.abs(x[0])**0.8 + 5 * np.sin(x[0]**3)) +\
         (np.abs(x[1])**0.8 + 5 * np.sin(x[1]**3))
    return np.array([-f1, -f2]) # maximum


def poloni_objectives(x):
    A1 = 0.5 * np.sin(1) - 2 * np.cos(1) + np.sin(2) - 1.5 * np.cos(2)
    A2 = 1.5 * np.sin(1) - np.cos(1) + 2 * np.sin(2) - 0.5 * np.cos(2)

    B1 = 0.5 * np.sin(x[0]) - 2 * np.cos(x[0]) + np.sin(x[1]) - 1.5 * np.cos(x[1])
    B2 = 1.5 * np.sin(x[0]) - np.cos(x[0]) + 2 * np.sin(x[1]) - 0.5 * np.cos(x[1])

    f1 = 1 + (A1 - B1)**2 + (A2 - B2)**2
    f2 = (x[0] + 3)**2 + (x[1] + 1)**2

    return np.array([-f1, -f2]) # maximum