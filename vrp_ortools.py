#!/usr/bin/python3
from cpmpy import *
import numpy as np
import math
from cpmpy.tools import ParameterTuner

"""
In the Vehicle Routing Problem (VRP), the goal is to find 
a closed path of minimal length
for a fleet of vehicles visiting a set of locations.

Or-tools specific version with its MultipleCircuit global constraint
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

# data
locations= [
        (288, 149), (288, 129), (270, 133), (256, 141), (256, 163),
        (236, 169), (228, 169), (228, 148), (220, 164)
]
n_city = len(locations)
depot = 0

# Pickup demand of each location 
demand = [0,3,4,8,8,10,5,3,9] # depot has no demand

# 3 vehicles and capacity of each vehicle 20
n_vehicle, q = 3, 20

# generate distances
distance_matrix = compute_euclidean_distance_matrix(locations)


s = SolverLookup.get("ortools")

# x[i,j] = 1 means that a vehicle goes from node i to node j 
x = boolvar(shape=distance_matrix.shape) 

# number of vehicles
s += sum(x[0,:]) <= n_vehicle

# VRP part (with global constraint)
## vehicle leaves and enter each node i exactly once (except depot at 0)
## no self visits (the 'if i!=j' part)
ort_arcs = [(i,j,b) for (i,j),b in np.ndenumerate(x) if i!=j]
s += DirectConstraint("AddMultipleCircuit", ort_arcs)


# capacity constraints, using cumulator intvar per node(city)
y = intvar(0, q, shape=n_city)
# depot takes no load
s += y[0] == 0
# if there is an arc, the dest value is source value + demand
s += [x[i,j].implies(y[j] == y[i] + demand[j]) for i in range(0,n_city) for j in range(1,n_city)]


# the objective is to minimze the travelled distance 
# sum(x*dist) does not work because 2D array, use .sum()
s.minimize((x*distance_matrix).sum())

val = s.solve()
print(s.status())

print("Total Cost of solution",int(s.objective_value()))
sol = x.value()
firsts = np.where(sol[0]==1)[0]
for i, dest in enumerate(firsts):
    msg = f"Vehicle {i}: 0"

    source = 0
    dist = 0
    while dest != 0:
        dist += distance_matrix[source,dest]
        msg += f" --{y[source].value()}--> {dest}[{demand[dest]}]"
        source = dest
        dest = np.argmax(sol[source])

    dist += distance_matrix[source,dest]
    msg += f" --{y[source].value()}--> {dest} :: total km={dist}"
    print(msg)

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
