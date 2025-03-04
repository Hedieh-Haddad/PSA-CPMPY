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

model = Model(
            sum(x*weights) <= capacity,
        maximize=
            sum(x*values)
        )

print("Value:", model.solve()) # solve returns objective value
print(f"Capacity: {capacity}, used: {sum(x.value()*weights)}")
items = np.where(x.value())[0]
print("In items:", items)
print("Values:  ", values[items])
print("Weights: ", weights[items])



tunables = {
    "search_branching": [0, 1, 2, 3, 4, 5, 6, 7],
    "linearization_level": [0, 1],
    'symmetry_level': [0, 1, 2]
}

defaults = {
    "search_branching": 1,
    "linearization_level": 1,
    'symmetry_level': 0
}

# Initialize the solver and tuner
solver = "ortools"
tuner = ParameterTuner(solver, model, tunables, defaults)
# Tune the parameters and get the best configuration
best_params= tuner.tune(time_limit=120)
best_runtime = tuner.best_runtime
# obj = tuner.obj if tuner.obj is not None else 0


print(best_params)
print(best_runtime)