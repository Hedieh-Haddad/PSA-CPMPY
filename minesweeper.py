#!/usr/bin/python3
"""
Minesweeper problem in CpMPy

Based on the Minesweeper model of Hakan Kjellerstrand
http://www.hakank.org/common_cp_models/#minesweeper

A specification is a square matrix of characters. Alphanumeric 
characters represent the number of mines adjacent to that field. 
X represent fields with an unknown number of mines adjacent to 
it (or an actual mine).
'''

E.g.
     "XX2X3X"
     "2XXXXX"
     "XX24X3"
     "1X34XX"
     "XXXXX3"
     "X3X3XX"

"""

from cpmpy import *
import numpy as np
from cpmpy.tools import ParameterTuner


X = -1
game = np.array([
            [2,3,X,2,2,X,2,1],
            [X,X,4,X,X,4,X,2],
            [X,X,X,X,X,X,4,X],
            [X,5,X,6,X,X,X,2],
            [2,X,X,X,5,5,X,2],
            [1,3,4,X,X,X,4,X],
            [0,1,X,4,X,X,X,3],
            [0,1,2,X,2,3,X,2]
            ])

rows, cols = game.shape
S = [-1,0,1] # for the neighbors of a cell

# Variables (mine or not)
mines = boolvar(shape=game.shape) 

model = Model()
for (r,c), val in np.ndenumerate(game):
    if val != X:
        # This cell cannot be a mine
        model += mines[r,c] == 0 
        # Count neighbors
        model += (sum(mines[r+a,c+b] for a in S for b in S \
                                     if r+a >= 0 and r+a < rows \
                                     and c+b >= 0 and c+b < cols) \
                  == val)


if model.solve():
    msg = "\n"
    for r,row in enumerate(game):
        for c,val in enumerate(row):
            if val != X:
                msg += str(val)
            elif mines[r,c].value():
                msg += "*"
            else:
                msg += " "
        msg += "\n"
    print(msg)
else:
    print("No solution.")

model.minimize(sum(mines))


tunables = {
    "search_branching": [0, 1, 2, 3, 4, 5, 6, 7],
    "linearization_level": [0, 1],
    'symmetry_level': [0, 1, 2]}

defaults = {
    "search_branching": 7,
    "linearization_level": 0,
    'symmetry_level': 1}

solver = "ortools"
tuner = ParameterTuner(solver, model, tunables, defaults)

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
