#!/usr/bin/python3
"""
The 'frietkot' problem; invented by Tias to explain SAT and SAT solving
http://homepages.vub.ac.be/~tiasguns/frietkot/
"""
from cpmpy import *
from cpmpy.tools import ParameterTuner

# Construct the model.
(mayo, ketchup, curry, andalouse, samurai) = boolvar(5)

# Pure CNF
Nora = mayo | ketchup
Leander = ~samurai | mayo
Benjamin = ~andalouse | ~curry | ~samurai
Behrouz = ketchup | curry | andalouse
Guy = ~ketchup | curry | andalouse
Daan = ~ketchup | ~curry | andalouse
Celine = ~samurai
Anton = mayo | ~curry | ~andalouse
Danny = ~mayo | ketchup | andalouse | samurai
Luc = ~mayo | samurai

allwishes = [Nora, Leander, Benjamin, Behrouz, Guy, Daan, Celine, Anton, Danny, Luc]

model = Model(allwishes)
if model.solve():
    print("Mayonaise = ", mayo.value())
    print("Ketchup = ", ketchup.value())
    print("Curry Ketchup = ", curry.value())
    print("Andalouse = ", andalouse.value())
    print("Samurai = ", samurai.value())
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

print("best_params", best_params)
print(best_runtime)
