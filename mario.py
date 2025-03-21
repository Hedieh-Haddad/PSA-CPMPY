#!/usr/bin/python3
"""
Mario problem in CPMpy

Based on the MiniZinc model, same data
"""
import numpy
from cpmpy import *
from cpmpy.tools import ParameterTuner
import numpy as np


np.random.seed(42)
data = { # a dictionary, json style
  'nbHouses': 100,  # Increased number of houses
  'MarioHouse': 1,
  'LuigiHouse': 2,
  'fuelMax': 20000,  # Increased fuel limit
  'goldTotalAmount': 10000,  # Increased total gold amount
  'conso': numpy.random.randint(0, 1000, size=(100, 100)).tolist(),  # Random fuel consumption values
  'goldInHouse': numpy.random.randint(0, 100, size=100).tolist(),  # Random gold values
}

# Python is offset 0, MiniZinc (source of the data) is offset 1
marioHouse, luigiHouse = data['MarioHouse']-1, data['LuigiHouse']-1 
fuelLimit = data['fuelMax']
nHouses = data['nbHouses']
arc_fuel = data['conso'] # arc_fuel[a,b] = fuel from a to b
arc_fuel = cpm_array(arc_fuel) # needed to do arc_fuel[var1] == var2

# s[i] is the house succeeding to the ith house (s[i]=i if not part of the route)
s = intvar(0,nHouses-1, shape=nHouses, name="s")

model = Model(
    #s should be a path, mimic (sub)circuit by connecting end-point back to start
    s[luigiHouse] == marioHouse,
    Circuit(s),  # should be subcircuit?
)

# consumption, knowing that always conso[i,i]=0 
# node_fuel[i] = arc_fuel[i, successor-of-i]
# observe how we do NOT create auxiliary CP variables here, just a list of expressions...
node_fuel = [arc_fuel[i, s[i]] for i in range(nHouses)]
model += sum(node_fuel) < fuelLimit

# amount of gold earned, only for stops visited, s[i] != i
gold = sum( (s != range(nHouses))*data['goldInHouse'] )
model.maximize(gold)

assert model.solve(), "Model is UNSAT!"
print("Gold:", gold.value()) # solve returns objective value
print("successor vars:",s.value())

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
