import numpy as np
import matplotlib.pyplot as plt
from pareto_cal import is_non_dominated_np

def plot_pareto_1d_2o(f, constraints, pareto, file_path):
    num_points = 1000  # adjust the number of generated points as needed
    x_values = np.linspace(constraints['X'][0], constraints['X'][1], num_points)
    function_values = np.array([f(x) for x in x_values])

    # Screening out the Pareto frontier
    pareto_real_mask = is_non_dominated_np(function_values)
    pareto_real = function_values[pareto_real_mask]

    # Custom colors
    tum_blue = '#0065BD'  # Converted from 0x0065BD
    tum_orange = '#E37222'   # Converted from 0xE37222
    tum_vory = '#DAD7CB'


    plt.scatter(-pareto_real[:, 0], -pareto_real[:, 1], label='True Pareto Points', color=tum_vory, alpha=0.5)
    
    # Plotting model-calculated Pareto points in red 'x'
    plt.scatter(-pareto[:, 0], -pareto[:, 1], label='Model Pareto Points', color=tum_blue, marker='x')

    plt.xlabel('Objective 1')
    plt.ylabel('Objective 2')
    plt.legend()
    # plt.show()
    plt.savefig(file_path, format='pdf')
    plt.close()

def plot_pareto_2d_2o(f, constraints, pareto, file_path):
    num_points = 100  # adjust the number of generated points as needed
    x_values_x = np.linspace(constraints['X0'][0], constraints['X0'][1], num_points)
    x_values_y = np.linspace(constraints['X1'][0], constraints['X1'][1], num_points)

    # Custom colors
    tum_blue = '#0065BD'  # Converted from 0x0065BD
    tum_orange = '#E37222'   # Converted from 0xE37222
    tum_vory = '#DAD7CB'

    import itertools
    points_array = np.array(list(itertools.product(x_values_x, x_values_y)))
    function_values = np.array([f(x.T) for x in points_array])

    # Screening out the Pareto frontier
    pareto_real_mask = is_non_dominated_np(function_values)
    pareto_real = function_values[pareto_real_mask]

    plt.scatter(-pareto_real[:, 0], -pareto_real[:, 1], label='True Pareto Points', color=tum_vory, alpha=0.5)
    
    # Plotting model-calculated Pareto points in red 'x'
    plt.scatter(-pareto[:, 0], -pareto[:, 1], label='Model Pareto Points', color=tum_blue, marker='x')

    plt.xlabel('Objective 1')
    plt.ylabel('Objective 2')
    plt.legend()
    # plt.show()
    plt.savefig(file_path, format='pdf')
    plt.close()

def plot_pareto_2d_3o(f, constraints, pareto, file_path):
    num_points = 100  # adjust the number of generated points as needed
    x_values_x = np.linspace(constraints['X0'][0], constraints['X0'][1], num_points)
    x_values_y = np.linspace(constraints['X1'][0], constraints['X1'][1], num_points)
    
    # Custom colors
    tum_blue = '#0065BD'  # Converted from 0x0065BD
    tum_orange = '#E37222'   # Converted from 0xE37222
    tum_vory = '#DAD7CB'

    import itertools
    points_array = np.array(list(itertools.product(x_values_x, x_values_y)))
    function_values = np.array([f(x.T) for x in points_array])

    # Screening out the Pareto frontier
    pareto_real_mask = is_non_dominated_np(function_values)
    pareto_real = function_values[pareto_real_mask]

    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plotting true Pareto points in blue
    ax.scatter(-pareto_real[:, 0], -pareto_real[:, 1], -pareto_real[:, 2], label='True Pareto Points', color=tum_vory, alpha=0.5)

    # Plotting model-calculated Pareto points in red 'x'
    ax.scatter(-pareto[:, 0], -pareto[:, 1], -pareto[:, 2], label='Model Pareto Points', color=tum_blue, marker='x')

    ax.set_xlabel('Objective 1')
    ax.set_ylabel('Objective 2')
    ax.set_zlabel('Objective 3')
    ax.legend()
    # plt.show()
    plt.savefig(file_path, format='pdf')
    plt.close()
