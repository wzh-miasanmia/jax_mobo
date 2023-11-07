# jax_mobo
## purpose
The purpose of this repo is to establish the use of JAX to complete a multi-objective Bayesian optimization problem and for solving the problem in the TENO-6 format in computational fluid dynamics.

## progress
The project is currently in its early stages and progress is as follows:

Completed:
- Gaussian process using JAX including the calculation of kernel, postrior and marginal likelihood function
- Implemented encapsulation: GaussianProcessRegressor, with methods like fit, optim, predict, etc.(not work with jax, think about it later)
- Single-objective Bayesian optimization

Plan to proceed:

- Multi-objective Bayesian optimization

## details
bayesian_jax:
Reference to the completed package on github has been changed to include the ability to draw 2d and 3d graphics

jax_learn:
Files to learn JAX

sobo_demo:
Code written by myself to complete the Bayesian optimization process using jax, robustness is very low, can't use too low a noise, problems with multidimensionality, still needs improvement


## how to use
```python
conda activate maenv
cd wzhmiasanmia/ma_workspace/jax_mobo/
```


