#!/usr/bin/python3
"""
N-queens problem in CPMpy

CSPlib prob054

Problem description from the numberjack example:
The N-Queens problem is the problem of placing N queens on an N x N chess
board such that no two queens are attacking each other. A queen is attacking
another if it they are on the same row, same column, or same diagonal.
"""

# load the libraries
import numpy as np
from cpmpy import *
from cpmpy.tools import ParameterTuner

def nqueens(N):
    # Variables (one per row)
    queens = intvar(1,N, shape=N, name="queens")

    # Constraints on columns and left/right diagonal
    m = Model([
        AllDifferent(queens),
        AllDifferent([queens[i] + i for i in range(N)]),
        AllDifferent([queens[i] - i for i in range(N)]),
    ])
    
    return (m, queens)

def nqueens_solve(N, prettyprint=True):
    (m, queens) = nqueens(N)

    if m.solve():
        print(m.status())

        if prettyprint:
            # pretty print
            line = '+---'*N+'+\n'
            out = line
            for queen in queens.value():
                out += '|   '*(queen-1)+'| Q '+'|   '*(N-queen)+'|\n'
                out += line
            print(out)
    else:
        print("No solution found")
    min_distance = min([abs(queens[i] - queens[j]) + abs(i - j) for i in range(N) for j in range(i + 1, N)])
    print("min_distance", min_distance)
    m.maximize(min_distance)
    return m


# if __name__ == "__main__":
N = 32
m = nqueens_solve(N)

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
    "time_evol": "Dynamic_Geometric"
}
user_params = {
    "init_round_type": "Dynamic",  # "Dynamic", "Static" , "None"
    "stop_type": "Timeout",  # "First_Solution" , "Timeout"
    "tuning_timeout_type": "Static",  # "Static" , "Dynamic", "None"
    "time_evol": "Dynamic_Geometric"  # "Static", "Dynamic_Geometric" , "Dynamic_Luby"
}

params = {**default_params, **user_params}

best_params = tuner.tune(
    time_limit=120,
    max_tries=30,
    **params
)
best_runtime = tuner.best_runtime

print(best_params)
print(best_runtime)