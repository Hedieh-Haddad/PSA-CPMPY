#!/usr/bin/python3
"""
Knapsack problem in CPMpy
 
Based on the Numberjack model of Hakan Kjellerstrand
"""
import numpy as np
from cpmpy import *
from cpmpy.tools import ParameterTuner

# Problem data
n = 10
np.random.seed(1)
values = np.random.randint(0,10, n)
weights = np.random.randint(1,5, n)
capacity = np.random.randint(sum(weights)*.2, sum(weights)*.5)

# Construct the model.
x = boolvar(shape=n, name="x")

m = Model(
            sum(x*weights) <= capacity,
        maximize=
            sum(x*values)
        )

print("Value:", m.solve()) # solve returns objective value
print(f"Capacity: {capacity}, used: {sum(x.value()*weights)}")
items = np.where(x.value())[0]
print("In items:", items)
print("Values:  ", values[items])
print("Weights: ", weights[items])




tunables = {
    "search_branching": [0, 1, 2, 3, 4, 5, 6, 7],
    "linearization_level": [0, 1],
    'symmetry_level': [0, 1, 2]}

defaults = {
    "search_branching": 7,
    "linearization_level": 0,
    'symmetry_level': 1}

solver = "ortools"
tuner = ParameterTuner(solver, m, tunables, defaults)

default_params = {
    "init_round_type": "Static",
    "stop_type": "Timeout",
    "tuning_timeout_type": "Static",
    "time_evol": "Dynamic_Geometric",
    "HPO": "Bayesian"
}
user_params = {
    "init_round_type": "Static",  # "Dynamic", "Static" , "None"
    "stop_type": "Timeout",  # "First_Solution" , "Timeout"
    "tuning_timeout_type": "Static",  # "Static" , "Dynamic", "None"
    "time_evol": "Static",  # "Static", "Dynamic_Geometric" , "Dynamic_Luby"
    "HPO": "Bayesian",  # "Hamming", "Bayesian", "Grid"
}

params = {**default_params, **user_params}

best_params = tuner.tune(
    time_limit=40,
    max_tries=10,
    **params
)
best_runtime = tuner.best_runtime

print(best_params)
print(best_runtime)
