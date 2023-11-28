# jax_mobo
## purpose
The purpose of this repo is to establish the use of JAX to complete a multi-objective Bayesian optimization problem and for solving the problem in the TENO-6 format in computational fluid dynamics.

## progress
The project is currently in its early stages and progress is as follows:

Completed:
- Gaussian process:
    - using JAX/numpy including the calculation of kernel, postrior and marginal likelihood function
    - Implemented encapsulation: GaussianProcessRegressor(class form) using JAX /numpy, with methods like fit, optim, predict, etc
- Single-objective Bayesian optimization:
    - with numpy
    - with JAX: do not have a proper minimize function
- Multi-objective Bayesian optimization:
    - with numpy
    - EHVI method: HV calculation using Dimension Sweep Algorithm
    - multi-objective Bayesian optimization with convergence checking

Plan to proceed:
- combined with jax-fluid numerical simulation


## details
bayesian_jax:
Reference to the completed package on github has been changed to include the ability to draw 2d and 3d graphics

sobo_demo:
Code written by zihao to complete the Bayesian optimization process using jax
noramlization problem still need to solve

mobo_learn:
learn how to implement mobo method, according to botorch

mobo_np:(main work)
Code written by zihao to complete the multi-objective bayesian optimization process using numpy


## usage
```python
conda activate maenv
cd ./jax_mobo/
```
- if you would like to have a test for sobo in numpy, try:
```python
cd ./jax_mobo/mobo_np
```
