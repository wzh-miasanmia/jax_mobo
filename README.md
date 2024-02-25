# Multi-Objective Optimization Branch

## Pipeline Description

### `GP_class.py`
Implements a Gaussian Process class based on numpy. It includes methods for fitting (`fit`), prediction (`predict`), and selecting the best parameters (`optim_np`).

### `mobo.py`
Defines the Expected Hypervolume Improvement (EHVI) function and the method for proposing new sampling points (`propose_location`), utilizing the L-BFGS-B algorithm to minimize the EHVI function. EHVI calculations are based on the hypervolume computation methods in `HV_cal`.

### `baseline` Folder
Contains results obtained from implosion (imp) and Taylor-Green Vortex (tgv) simulations using TensorFlow. A Jupyter notebook in this folder records the objective function values obtained from baseline calculations, which are directly utilized in `simulation.py`.

### `imp` and `tgv` Folders
Store initial settings and results related to numerical simulations, including initial conditions, objective function calculation methods, and all simulation outcomes. Given the extensive optimization process and the large number of sampling points, the results folders can become quite large.

### `precious_results`
Contains some previously obtained results, which are not essential and can be ignored.

### `simulation.py`
Defines the interface for the objective function, numerical simulation module, and optimization module. For details, see the objectives description section.

### `optim_process_mobo_new.py`
The primary optimization script, outlining the entire optimization process. It modifies the rule for selecting reference points compared to `optim_process_mobo` and provides two methods for obtaining initial points. It starts by initializing a GP model, then selects some initial points randomly as CFD parameters, which are input into the simulation to obtain initial objective values (`y` values) for updating the GP model. Note: If using the same settings and objective functions, previously obtained results can serve as initial points, avoiding the need to restart the optimization process due to interruptions. The optimization loop begins by proposing new sampling points (using `propose_location` defined in `sobo`), inputting them into the simulation for results, and adding the new points (`x_new`, `y_new`) to update the GP model, continuing until reaching a specified number of iterations or other optimization criteria are met.

### `run_OP.py`
Serves as a template for using the entire process pipeline. Before optimizing, it's necessary to define `SCHEME`, `scheme_setup`, `noise`, `n_init`, and `n_iter`.

## Objectives Description
- `f_disper`: Represents Limited Dispersion, derived from 2D implosion (imp).
- `f_tke`: Represents Turbulent Kinetic Energy, derived from Taylor-Green Vortex (tgv).
- `f_cons`: Represents Kinetic Energy at the terminal time, derived from Taylor-Green Vortex (tgv).

## Post-Processing
- `visualization.ipynb`: The primary post-processing script. Inputs optimization results (`results.csv`), selects optimal points through Pareto frontier definition, and calculates the best points using Cross-Validation (CV).
- `compare_results.ipynb`: Compares a series of result points using CV.
- `plot_GP_fix.ipynb`: Fixes a certain input to visualize the relationship between the three objective functions and the input in a two-dimensional space through the GP model.

## Auxiliary Files
- `calculate_initial_points.ipynb`: Calculates objective function values from points that have undergone numerical simulation.
- `delete_folders.ipynb`: Quickly deletes the `results` folders in `imp` and `tgv` to reduce storage usage.
