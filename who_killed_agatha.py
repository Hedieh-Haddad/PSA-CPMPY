#!/usr/bin/python3
"""
Who killed agatha? problem in CPMpy

Based on the Numberjack model of Hakan Kjellerstrand
see also: http://www.hakank.org/constraint_programming_blog/2014/11/decision_management_community_november_2014_challenge_who_killed_agath.html
"""
from cpmpy import *
from enum import Enum
import numpy
from cpmpy.tools import ParameterTuner


# Agatha, the butler, and Charles live in Dreadsbury Mansion, and 
# are the only ones to live there. 
n = 3
(agatha, butler, charles) = range(n) # enum constants
names = ["Agatha herself", "the butler", "Charles"] # reverse mapping

# Who killed agatha?
victim = agatha
killer = intvar(0,2, name="killer")

hates  = boolvar(shape=(n,n), name="hates")
richer = boolvar(shape=(n,n), name="richer")

model = Model(
    # A killer always hates, and is no richer than, his victim. 
    # note; 'killer' is a variable, so must write ==1/==0 explicitly
    hates[killer, victim] == 1,
    richer[killer, victim] == 0,

    # implied richness: no one richer than himself, and anti-reflexive
    [~richer[i,i] for i in range(n)],
    [(richer[i,j]) == (~richer[j,i]) for i in range(n) for j in range(i+1,n)],

    # Charles hates noone that Agatha hates. 
    [(hates[agatha,i]).implies(~hates[charles,i]) for i in range(n)],

    # Agatha hates everybody except the butler. 
    hates[agatha,(agatha,charles,butler)] == [1,1,0],

    # The butler hates everyone not richer than Aunt Agatha. 
    [(~richer[i,agatha]).implies(hates[butler,i]) for i in range(n)],

    # The butler hates everyone whom Agatha hates. 
    [(hates[agatha,i]).implies(hates[butler,i]) for i in range(n) ],

    # Noone hates everyone. 
    [sum(hates[i,:]) <= 2 for i in range(n)],
)

# Solve and print
if model.solve():
    print("Who killed Agatha? It was...", names[killer.value()])
else:
    print("No solution found")

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
