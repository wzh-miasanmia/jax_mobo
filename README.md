# Project Overview
## Purpose
The purpose of this repo is to establish the use of Jax and numpy to complete a multi-objective Bayesian optimization problem and for choosding the hyperparameters in the TENO6-DV format in computational fluid dynamics (based on JAX-fluids).
## Description
This project is divided into three main parts, each residing in its own separate branch:

- **mobo branch**: This branch contains the implementation of Multi-Objective Bayesian Optimization (MOBO) using two different approaches: JAX and NumPy. In the `mobo_np` directory, there are some simple test functions used to verify the correctness of the algorithm.

- **optim_single branch**: This branch focuses on optimizing a single objective, specifically targeting the result calculation of `f_tke` from TGV (Taylor-Green Vortex) simulations.

- **optim_multi branch**: Dedicated to multi-objective optimization, this branch deals with several objectives including `f_disper`, `f_tke`, and `f_cons`.

Each branch is tailored to address specific aspects of optimization problems, from single-objective to multi-objective scenarios, employing various computational methods and verification tests to ensure the efficacy and accuracy of the optimization algorithms employed.


## Project Setup Guide

This guide will walk you through the process of setting up your Python environment and installing the required packages for this project. We'll be using `conda`, a powerful package manager and environment management system, which makes it easy to install, run, and update packages and their dependencies.

### Prerequisites

Before you begin, ensure you have `conda` installed on your system. If you do not have `conda` installed, you can download and install Miniconda, which is a smaller, lightweight version of Anaconda. Visit [Miniconda's official site](https://docs.conda.io/en/latest/miniconda.html) for installation instructions.

### Step 1: Clone the Project Repository

First, clone the project repository to your local machine using Git. Open your terminal and run the following command:

```bash
git clone https://github.com/wzh-miasanmia/jax_mobo.git
cd jax_mobo
```

### Step 2: Create a New Conda Environment
Create a new conda environment specifically for this project. This ensures that the project's dependencies do not interfere with other Python projects on your system.

```bash
conda create --name your_env_name python=3.8
```
Replace your_env_name with a name of your choice for the new environment, and python=3.8 with the Python version required by your project.

### Step 3: Activate the Conda Environment
Activate the newly created conda environment:

```bash
conda activate your_env_name
```

### Step 4: Install Required Packages
To install the required packages, navigate to the directory containing the requirements.txt file you generated earlier, and run:
```bash
conda install --file requirements.txt
```
### Step 5: Verify Installation
After installation, you can verify that all packages were installed correctly by listing the installed packages:

### Troubleshooting
If you encounter any errors during the installation of packages, try searching for the error message online for specific solutions, or consult the official documentation of the problematic package.
Ensure that your conda is up to date by running conda update conda.
For issues related to pip packages, ensure that your pip is up to date by running pip install --upgrade pip in your conda environment.