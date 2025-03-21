#!/usr/bin/python3
"""
Send more money in CPMpy

   SEND
 + MORE
 ------
  MONEY

"""
from cpmpy import *
import numpy as np
from cpmpy.tools import ParameterTuner

# Construct the model.
s,e,n,d,m,o,r,y,a,b,c,f,g,h,i,j,k,l,p,q,t,u,v,w,x,z = intvar(0,9, shape=26)

model = Model(
    AllDifferent([s,e,n,d,m,o,r,y,a,b,c,f,g,h,i,j,k,l,p,q,t,u,v,w,x,z]),
    (    sum(   [s,e,n,d] * np.array([       1000, 100, 10, 1]) ) \
       + sum(   [m,o,r,e] * np.array([       1000, 100, 10, 1]) ) \
      == sum( [m,o,n,e,y] * np.array([10000, 1000, 100, 10, 1]) ) ),
    s > 0,
    m > 0,
    # Additional constraints
    a + b == c,
    f + g == h,
    i + j == k,
    l + m == n,
    o + p == q,
    r + s == t,
    u + v == w,
    x + y == z,
)

print(model)
print("") # blank line

# Solve and print
if model.solve():
    print("  S,E,N,D =   ", [x.value() for x in [s,e,n,d]])
    print("  M,O,R,E =   ", [x.value() for x in [m,o,r,e]])
    print("M,O,N,E,Y =", [x.value() for x in [m,o,n,e,y]])
else:
    print("No solution found")
objective = sum([s, e, n, d, m, o, r, y, a, b, c, f, g, h, i, j, k, l, p, q, t, u, v, w, x, z])
model.minimize(objective)

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
    "init_round_type": "Dynamic",  # "Dynamic", "Static" , "None"
    "stop_type": "Timeout",  # "First_Solution" , "Timeout"
    "tuning_timeout_type": "Static",  # "Static" , "Dynamic", "None"
    "time_evol": "Static",  # "Static", "Dynamic_Geometric" , "Dynamic_Luby"
    "HPO": "Hamming",  # "Hamming", "Bayesian", "Grid"
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
