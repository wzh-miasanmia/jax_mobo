import numpy as np

def calculate_hypervolume(data, reference_point):
    """
    Calculate hypervolume indicator.

    Parameters:
    - data: A numpy array containing multiple objective vectors. Each row represents a solution's objective vector.
    - reference_point: Reference point with the same dimensionality as the objective vectors in the data.

    Returns:
    - hypervolume: Hypervolume indicator value.
    """

    # Check the validity of input data
    if len(data) == 0 or len(data[0]) != len(reference_point):
        raise ValueError("Invalid data or reference point dimensions.")

    # Calculate the non-dominated set
    non_dominated_set = get_non_dominated_set(data)

    # Initialize the hypervolume indicator
    hypervolume = 0.0

    # Calculate the hypervolume
    for solution in non_dominated_set:
        hypervolume += np.prod(np.maximum(0, reference_point - solution))

    return hypervolume

def get_non_dominated_set(data):
    """
    Get the non-dominated set from the given data.

    Parameters:
    - data: A numpy array containing multiple objective vectors. Each row represents a solution's objective vector.

    Returns:
    - non_dominated_set: Non-dominated set.
    """

    non_dominated_set = []

    for solution in data:
        if is_dominated(solution, non_dominated_set):
            continue
        non_dominated_set = remove_dominated_solutions(solution, non_dominated_set)
    print(non_dominated_set)
    return non_dominated_set

def is_dominated(solution, solution_set):
    """
    Check if the given solution is dominated by any solution in the set.

    Parameters:
    - solution: Objective vector of the solution to be checked.
    - solution_set: Set of solutions containing multiple objective vectors.

    Returns:
    - True if the solution is dominated by any solution in the solution_set, otherwise False.
    """

    for existing_solution in solution_set:
        if all(existing_solution <= solution) and any(existing_solution < solution):
            return True

    return False

def remove_dominated_solutions(solution, solution_set):
    """
    Remove solutions dominated by the given solution from the solution set.

    Parameters:
    - solution: Objective vector of the solution to be checked.
    - solution_set: Set of solutions containing multiple objective vectors.

    Returns:
    - Solution set after removing dominated solutions.
    """

    filtered_set = [existing_solution for existing_solution in solution_set if not all(existing_solution <= solution)]
    filtered_set.append(solution)

    return filtered_set

# Sample data and reference point
data = np.array([[3, 4], [1, 6], [5, 2], [2, 5]])
reference_point = np.array([6, 7])
# Calculate the hypervolume indicator
hypervolume = calculate_hypervolume(data, reference_point)

print("Hypervolume:", hypervolume)
