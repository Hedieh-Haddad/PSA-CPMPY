#!/usr/bin/python3
from cpmpy import *
import numpy as np
import math
from cpmpy.tools import ParameterTuner

"""
Taken from Google Ortools example https://developers.google.com/optimization/routing/tsp
  
The Traveling Salesman Problem (TSP) is stated as follows.
Let a directed graph G = (V, E) be given, where V = {1, ..., n} is
a set of nodes, E <= V x V is a set of arcs. Let also each arc
e = (i,j) be assigned a number c[i,j], which is the length of the
arc e. The problem is to find a closed path of minimal length going
through each node of G exactly once.
"""
def compute_euclidean_distance_matrix(locations):
    """Computes distances between all points (from ortools docs)."""
    n_city = len(locations)
    distances = np.zeros((n_city,n_city))
    for from_counter, from_node in enumerate(locations):
        for to_counter, to_node in enumerate(locations):
            if from_counter != to_counter:
                distances[from_counter][to_counter] = (int(
                    math.hypot((from_node[0] - to_node[0]),
                               (from_node[1] - to_node[1]))))
    return distances.astype(int)

# locations= [
#     (288, 149), (288, 129), (270, 133), (256, 141), (256, 163), (246, 157),
#     (236, 169), (228, 169), (228, 148), (220, 164), (212, 172), (204, 159)
# ]
np.random.seed(42)  # For reproducibility
num_locations = 100  # Try 500 for an even harder problem!
locations = [(np.random.randint(0, 1000), np.random.randint(0, 1000)) for _ in range(num_locations)]

distance_matrix = compute_euclidean_distance_matrix(locations)
n_city = len(locations)


# we use the successor variable formulation and circuit global constraint here
# alternative is to model like in vrp.py

# x[i]=j means that j is visited immediately after i
x = intvar(0, n_city-1, shape=n_city)

# The 'circuit' global constraint ensures that the successor variables from a circuit
model = Model( Circuit(x) )

# the objective is to minimze the travelled distance 
distance_matrix = cpm_array(distance_matrix) # for indexing with variable
travel_distance = sum(distance_matrix[i, x[i]] for i in range(n_city))
model.minimize(travel_distance)

# print(model)

model.solve()
print(model.status())

print("Total Cost of solution", travel_distance.value())
def display(sol):
    x = 0
    msg = "0"
    while sol[x] != 0:
        x = sol[x]
        msg += f" --> {x}"
    print(msg + " --> 0")
display(x.value())



tunables = {}

defaults = {}

solver = ""
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
    "HPO": "Bayesian",  # "Hamming", "Bayesian", "Grid"
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
