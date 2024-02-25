# Optimization Branch with a Single Objective Function: TGV

## Description

This repository contains several key components designed for optimization using Gaussian Processes and the Expected Improvement (EI) strategy, specifically tailored for problems with a single objective function.

### `GP_class.py`
Implements a Gaussian Process class based on NumPy. It includes methods for fitting the model (`fit`), making predictions (`predict`), and optimizing hyperparameters using a custom optimization method (`optim_np`).

### `sobo.py`
Defines the Expected Improvement (EI) function and the selection process for new sampling points. It utilizes the L-BFGS-B algorithm to find the minimum of the EI function, facilitating efficient sampling in the optimization process.

### `simulation.py`
Specifies the target function, numerical simulation modules, and interfaces for the optimization process. This component is crucial for defining the problem space and simulating the effects of various input parameters.

### `optim_process_sobo.py`
The primary optimization script, detailing the entire optimization workflow:
- Initializes a Gaussian Process (GP) model.
- Randomly selects initial points to serve as the starting parameters for the Computational Fluid Dynamics (CFD) simulations.
- Performs numerical simulations via the `simulation` module to obtain the initial `y` values associated with the starting points.
- Updates the GP model with these initial points.
- Enters an optimization loop where it:
  - Employs `propose_location` from `sobo.py` to select the next sampling point.
  - Passes this point through the `simulation` module to obtain results.
  - Adds the new point (`x_new`, `y_new`) to the model, updating the GP model accordingly.
- This loop continues until a specified number of iterations or other optimization criteria are met.

### `test.ipynb`
Serves as a template for using the optimization pipeline. Before starting the optimization, it is necessary to define several parameters including `SCHEME`, `scheme_setup`, `noise`, `n_init`, and `n_iter`.
