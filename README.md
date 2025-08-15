# Seasonal Predator-Prey System Simulations
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

This repository contains scripts to simulate and analyze the dynamics of a predator-prey system using ordinary differential equations (ODEs). The project is based on the work of [Tyson et al. (2016)](https://doi.org/10.1086/688665) and adapted for Rodent-Mustelid system in Fennoscandia. The repository includes tools for solving and visualizing the system's behavior under varying seasonal conditions. Project is still in progress and being updated. 

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Scripts](#scripts)
- [License](#license)


## Project Overview

This project models the interaction between prey (rodents) and predators (mustelids) using a system of ODEs. The model incorporates seasonal variations via different predatory responce and specifically focuses on the length of the summer season. The primary aim is to generate Poincaré sections to analyze the system's dynamics under different seasonal lengths.

## Requirements

To run the scripts in this repository, you need to have Python installed along with the following packages:

- `numpy >= 1.20`          # Numerical operations and arrays
- `matplotlib >= 3.4`     # Plotting and visualization
- `numba >= 0.53`          # JIT acceleration for numerical functions

The standard library modules csv and os are also used, but you do not need to install them separately.

### Installation

To clone this repository, follow these steps:

1. Open your terminal or command prompt.
2. Navigate to the directory where you want to clone the repository.
3. Run the following command:
```bash
git clone https://github.com/Jullcifer/predator-prey-seasonality.git
```

To install all the dependencies:

```bash
pip install numpy>=1.20 matplotlib>=3.4 numba>=0.53
```
Or, using a requirements.txt file, run:

```bash
pip install -r requirements.txt
```

## Usage

Each script in this repository serves a specific purpose in the simulation and analysis of the rodent-mustelid system. Below is a brief description of each script and how to use them.

### Model Specification and Input Data

To use the methods for analysing your own equations/system of ODEs, you need to change the equations given in the `seasonal_odes.py` script. Change both the parameter values and the equations, if necessary. Make sure that the new parameters are then passed on the `rk4_solver.py`. To run the analysis for rodent-mustelid system, you don't need to change the equations shape, but you can change the parameter values if desired. No input data is needed in this case.

### Run order to produce main plots

To make a 2D bifurcation diagram run:
1. `main_clusteringsimulation.py`
2. `main_bifurcationsimulation.py`

Without making any changes to the code, running these scripts should produce the following test bifurcation plot:
<img width="700" alt="BifurcationGridPlot" src="https://github.com/user-attachments/assets/28126f18-e0cc-41f1-981e-a427ef0bd356" />


To make Poincare plots for selected values of season length run (you can change the values of summer lengths for plots in the same script, if desired):
1. `plot_poincare_sections.py`

Without making any changes to the code, running this script should produce 10 Poincare maps, including following (#1 and #7):<img width="500" alt="plot_1" src="https://github.com/user-attachments/assets/57bb9c6f-f0f9-4206-961b-12991fae9637" />
  <img width="500" alt="plot_7" src="https://github.com/user-attachments/assets/2e14cde8-4d91-4795-a092-2eea657a8afe" />

## Scripts

### rk4_solver.py
This script provides a Runge-Kutta 4th order solver for solving ODEs. It is optimized using numba for performance and can simulate for a matrix of initial conditions for prey (`n`) and predator (`p`).

### seasonal_odes.py
This script defines the system of ODEs for the rodent-mustelid dynamics, adapted from [Tyson et al. (2016)](https://doi.org/10.1086/688665). It utilizes parameters from the rodent-mustelid system in Fennoscandia.

### plot_poincare_sections.py
This script solves the seasonal ODEs using the Runge-Kutta 4th order method and plots Poincaré sections for various lengths of the summer season (`a_s`). It utilizes the `rk4_solver` to perform numerical integration and `seasonal_odes` to define the system of equations.

### main_clusteringsimulation.py
This script generates clustered Poincaré maps for a range of seasonal lengths (`a_s`) and values of density-dependence (`nu`). It performs simulations for each combination of aS and nu, saves the results in organized folders, and outputs the following:

- Poincaré Section Plots;
- CSV Files: Lists of classified behaviors (e.g., chaos, cycles, quasi-periodic orbits);
- FTLE Values: Lyapunov exponents for analyzing system stability.

### main_bifurcationsimulation.py
This script generates a bifurcation diagram to classify the system's behavior across a grid of parameter values. It categorizes each grid point into one of six behavioral types. The script outputs:

- category_matrix.txt: A matrix of classifications for each grid point;
- Bifurcation diagram plot.

## License
This project is licensed under the MIT License. See the LICENSE file for details.



