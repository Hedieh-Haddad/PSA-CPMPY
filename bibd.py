#!/usr/bin/python3
"""
Balanced Incomplete Block Design (BIBD) in CPMpy

CSPlib prob028

Problem description from the numberjack example:
A BIBD is defined as an arrangement of v distinct objects into b blocks such
that each block contains exactly k distinct objects, each object occurs in
exactly r different blocks, and every two distinct objects occur together in
exactly lambda blocks.
Another way of defining a BIBD is in terms of its
incidence matrix, which is a v by b binary matrix with exactly r ones per row,
k ones per column, and with a scalar product of lambda 'l' between any pair of
distinct rows.
"""

# load the libraries
import numpy as np
from cpmpy import *
from cpmpy.tools import ParameterTuner


# Data
v, b = 11, 11
r, k = 5, 5
l = 2

# Variables, incidence matrix
block = boolvar(shape=(v,b), name="block")

# Constraints on incidence matrix
m = Model(
        [sum(row) == r for row in block],
        [sum(col) == k for col in block.T],
)

# the scalar product of every pair of distinct rows sums up to `l`
for row_a in range(v):
    for row_b in range(row_a+1,v):
        m += sum(block[row_a,:] * block[row_b,:]) == l


if m.solve():
    # pretty print
    print(f"BIBD: {b} obj, {v} blocks, r={r}, k={k}, l={l}")
    for (i,row) in enumerate(block.value()):
        srow = "".join('X ' if e else '  ' for e in row)
        print(f"Object {i+1}: [ {srow}]")
else:
    print("No solution found")

m.minimize(b)

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
    "time_evol": "Dynamic_Geometric"}
user_params = {
    "init_round_type": "Dynamic",  # "Dynamic", "Static" , "None"
    "stop_type": "Timeout",  # "First_Solution" , "Timeout"
    "tuning_timeout_type": "Static",  # "Static" , "Dynamic", "None"
    "time_evol": "Static" # "Static", "Dynamic_Geometric" , "Dynamic_Luby"
}

params = {**default_params, **user_params}

best_params = tuner.tune(
    time_limit=120,
    max_tries=10,
    **params
)
best_runtime = tuner.best_runtime

print(best_params)
print(best_runtime)
