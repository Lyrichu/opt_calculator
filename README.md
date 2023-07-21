# Multi-Algorithm Optimization GUI Application

This application provides an interface for optimizing functions using a variety of algorithms including Powell, CG, BFGS, L-BFGS-B, TNC, COBYLA, SLSQP, trust-constr, Genetic Algorithm (GA) and Particle Swarm Optimization (PSO). It provides options to customize the parameters of the chosen algorithm.

## Features

1. Function Input: This GUI accepts a function written in python syntax that you wish to optimize. It also supports multiple variables.

2. Variable Input: Define the variables used in the function in a separate input box.

3. Bounds Input: Define the lower and upper bounds for each variable.

4. Optimization Algorithm Selection: Choose from a variety of optimization algorithms in the dropdown menu.

5. Algorithm Parameters: When GA or PSO is chosen, a grid of input boxes appears allowing the user to customize the parameters of the algorithm.

## Optimization Algorithms

1. Powell, CG (Conjugate Gradient), BFGS (Broyden-Fletcher-Goldfarb-Shanno), L-BFGS-B, TNC (Truncated Newton), COBYLA (Constrained Optimization BY Linear Approximations), SLSQP (Sequential Least Squares Programming) and trust-constr are traditional gradient-based or direct search optimization methods with different advantages and limitations.

2. GA (Genetic Algorithm): Genetic algorithms are inspired by the process of natural selection and are used to generate solutions to optimization and search problems. They use bio-inspired operators such as mutation, crossover and selection.

3. PSO (Particle Swarm Optimization): PSO is a computational method that optimizes a problem by iteratively trying to improve a candidate solution with regard to a given measure of quality, mimicking the social behavior of bird flocking or fish schooling.

## How to Use

1. Enter your function in the Function Input box.
2. Define the variables used in your function in the Variables Input box, separated by commas.
3. Define the lower and upper bounds for each variable in the Bounds Input box, separated by commas.
4. Choose your desired optimization algorithm from the dropdown menu.
5. If GA or PSO is chosen, input or adjust the parameters for the chosen algorithm.
6. Press the "Start Optimization" button to run the optimization.

## Note

The tooltips on each parameter input box provide information about the parameter they are linked to.

## Requirements

To run this application, you need PySide6 and DEAP installed. They can be installed using pip:
```shell
pip install pyside6 deap
