# Seasonal Predator-Prey System Simulations

This repository contains scripts to simulate and analyze the dynamics of a predator-prey system using ordinary differential equations (ODEs). The project is based on the work of [Tyson et al. (2016)](https://doi.org/10.1086/688665) and adapted for Rodent-Mustelid system in Fennoscandia. The repository includes tools for solving and visualizing the system's behavior under varying seasonal conditions. Project is still in progress and being updated. 

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Scripts](#scripts)


## Project Overview

This project models the interaction between prey (rodents) and predators (mustelids) using a system of ODEs. The model incorporates seasonal variations via different predatory responce and specifically focuses on the length of the summer season. The primary aim is to generate Poincaré sections to analyze the system's dynamics under different seasonal lengths.

## Requirements

To run the scripts in this repository, you need to have Python installed along with the following packages:

- numpy >= 1.20          # Numerical operations and arrays
- matplotlib >= 3.4     # Plotting and visualization
- numba >= 0.53          # JIT acceleration for numerical functions

The standard library modules csv and os are also used, but you do not need to install them separately.

### Installation
You can install all required packages with:

```bash
pip install numpy>=1.20 matplotlib>=3.4 numba>=0.53
```
Or, using a requirements.txt file, run:

```bash
pip install -r requirements.txt
```

To clone this repository, follow these steps:

1. Open your terminal or command prompt.
2. Navigate to the directory where you want to clone the repository.
3. Run the following command:
```bash
git clone https://github.com/Jullcifer/predator-prey-seasonality.git
```

## Usage

Each script in this repository serves a specific purpose in the simulation and analysis of the rodent-mustelid system. Below is a brief description of each script and how to use them.

### Model Specification and Input Data

To use the methods for analysing your own equations/system of ODEs, you need to change the equations given in the seasonal_odes.py script. Change both the parameter values and the equations, if necessary. Make sure that the new parameters are then passed on the rk4_solver.py. To run the analysis for rodent-mustelid system, you don't need to change the equations shape, but you can change the parameter values if desired. No input data is needed in this case.

### Run order to produce main plots

To make a 2D bifurcation diagram run:
1. main_clusteringsimulation.py
2. main_bifurcationsimulation.py

To make Poincare plots for selected values of season length run (you can change the values of summer lengths for plots in the same script, if desired):
1. plot_poincare_sections.py

## Scripts

### rk4_solver.py
This script provides a Runge-Kutta 4th order solver for solving ODEs. It is optimized using numba for performance and can simulate for a matrix of initial conditions for prey (n) and predator (p).

### seasonal_odes.py
This script defines the system of ODEs for the rodent-mustelid dynamics, adapted from [Tyson et al. (2016)](https://doi.org/10.1086/688665). It utilizes parameters from the rodent-mustelid system in Fennoscandia.

### plot_poincare_sections.py
This script solves the seasonal ODEs using the Runge-Kutta 4th order method and plots Poincaré sections for various lengths of the summer season (`a_s`). It utilizes the `rk4_solver` to perform numerical integration and `seasonal_odes` to define the system of equations.

### main_clusteringsimulation.py
The main function for creating multiple grid simulations of the clustered poincare maps.





