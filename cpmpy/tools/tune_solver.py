"""
    This file implements parameter tuning for constraint solvers based on SMBO and using adaptive capping.
    Based on the following paper:
    Ignace Bleukx, Senne Berden, Lize Coenen, Nicholas Decleyre, Tias Guns (2022). Model-Based Algorithm
    Configuration with Adaptive Capping and Prior Distributions. In: Schaus, P. (eds) Integration of Constraint
    Programming, Artificial Intelligence, and Operations Research. CPAIOR 2022. Lecture Notes in Computer Science,
    vol 13292. Springer, Cham. https://doi.org/10.1007/978-3-031-08011-1_6

    - DOI: https://doi.org/10.1007/978-3-031-08011-1_6
    - Link to paper: https://rdcu.be/cQyWR
    - Link to original code of paper: https://github.com/ML-KULeuven/DeCaprio

    This code currently only implements the author's 'Hamming' surrogate function.
    The parameter tuner iteratively finds better hyperparameters close to the current best configuration during the search.
    Searching and time-out start at the default configuration for a solver (if available in the solver class)
"""
import time
from random import shuffle

import numpy as np

from ..solvers.utils import SolverLookup, param_combinations
from ..solvers.solver_interface import ExitStatus
from cpmpy.tools.PSA import *


class ParameterTuner:
    """
        Parameter tuner based on DeCaprio method [ref_to_decaprio]
    """

    def __init__(self, solvername, model, all_params=None, defaults=None):
        """            
            :param solvername: Name of solver to tune
            :param model: CPMpy model to tune parameters on
            :param all_params: optional, dictionary with parameter names and values to tune. If None, use predefined parameter set.
        """
        self.solvername = solvername
        self.model = model
        self.all_params = all_params
        self.best_params = defaults
        if self.all_params is None:
            self.all_params = SolverLookup.lookup(solvername).tunable_params()
            self.best_params = SolverLookup.lookup(solvername).default_params()
        print(self.all_params)
        print(self.best_params)

    def tune(self, time_limit=None, max_tries=None, fix_params={}, **kwargs):
        """
            :param time_limit: Time budget to run tuner in seconds. Solver will be interrupted when time budget is exceeded
            :param max_tries: Maximum number of configurations to test
            :param fix_params: Non-default parameters to run solvers with.
        """
        best_params, best_runtime = Probe(self.solvername, self.model, time_limit, max_tries, self.all_params, self.best_params, fix_params, **kwargs).get_best_params_and_runtime()
        # best_params, best_runtime = Probe(self.solvername, self.model, time_limit, max_tries, HPO, self.all_params, self.best_params, fix_params).get_best_params_and_runtime()
        # best_params, best_runtime = probe_instance.get_best_params_and_runtime()
        self.best_runtime = best_runtime
        return best_params




class GridSearchTuner(ParameterTuner):

    def __init__(self, solvername, model, all_params=None, defaults=None):
        super().__init__(solvername, model, all_params, defaults)
    def tune(self, time_limit=None, max_tries=None, fix_params={}):
        """
            :param: time_limit: Time budget to run tuner in seconds. Solver will be interrupted when time budget is exceeded
            :param: max_tries: Maximum number of configurations to test
            :param: fix_params: Non-default parameters to run solvers with.
        """
        if time_limit is not None:
            start_time = time.time()

        # Init solver
        solver = SolverLookup.get(self.solvername, self.model)
        solver.solve(**self.best_params)


        self.base_runtime = solver.status().runtime
        self.best_runtime = self.base_runtime

        # Get all possible hyperparameter configurations
        combos = list(param_combinations(self.all_params))
        shuffle(combos) # test in random order

        if max_tries is not None:
            combos = combos[:max_tries]

        for params_dict in combos:
            # Make new solver
            solver = SolverLookup.get(self.solvername, self.model)
            # set fixed params
            params_dict.update(fix_params)
            timeout = self.best_runtime
            # set timeout depending on time budget
            if time_limit is not None:
                timeout = min(timeout, time_limit - (time.time() - start_time))
            # run solver
            solver.solve(**params_dict, time_limit=timeout)
            if solver.status().exitstatus == ExitStatus.OPTIMAL and solver.status().runtime < self.best_runtime:
                self.best_runtime = solver.status().runtime
                # update surrogate
                self.best_params = params_dict

            if time_limit is not None and (time.time() - start_time) >= time_limit:
                break

        return self.best_params



