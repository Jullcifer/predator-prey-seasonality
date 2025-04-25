# predator_prey_seasonality
# Project Title: Seasonal Predator-Prey System Simulations

This repository contains scripts to simulate and analyze the dynamics of a predator-prey system using ordinary differential equations (ODEs). The project is based on the work of [Tyson et al. (2016)](https://doi.org/10.1086/688665) and adapted for Rodent-Mustelid system in Fennoscandia. The repository includes tools for solving and visualizing the system's behavior under varying seasonal conditions. Project is still in progress and being updated.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Scripts](#scripts)


## Project Overview

This project models the interaction between prey (rodents) and predators (mustelids) using a system of ODEs. The model incorporates seasonal variations via different predatory responce and specifically focuses on the length of the summer season. The primary aim is to generate Poincaré sections to analyze the system's dynamics under different seasonal lengths.

## Installation

To run the scripts in this repository, you need to have Python installed along with the following packages:

- `numpy`
- `matplotlib`
- `numba`

You can install these packages using pip:

```bash
pip install numpy matplotlib numba
```

## Usage

Each script in this repository serves a specific purpose in the simulation and analysis of the rodent-mustelid system. Below is a brief description of each script and how to use them.

## Scripts

### rk4_solver.py
This script provides a Runge-Kutta 4th order solver for solving ODEs. It is optimized using numba for performance and can simulate for a matrix of initial conditions for prey (n) and predator (p).

### seasonal_odes.py
This script defines the system of ODEs for the rodent-mustelid dynamics, adapted from [Tyson et al. (2016)](https://doi.org/10.1086/688665). It utilizes parameters from the rodent-mustelid system in Fennoscandia.

### plot_poincare_sections.py
This script solves the seasonal ODEs using the Runge-Kutta 4th order method and plots Poincaré sections for various lengths of the summer season (`a_s`). It utilizes the `rk4_solver` to perform numerical integration and `seasonal_odes` to define the system of equations.

#### Dependencies

- `numpy`
- `matplotlib`
- `rk4_solver` (for `rk4solver`)
- `seasonal_odes` (for `np_odes`)

### main_clusteringsimulation.py
The main function for creating multiple grid simulations of the clustered poincare maps.




