import time
from random import shuffle
from skopt import Optimizer
from skopt.utils import point_asdict, dimensions_aslist
import numpy as np
from ..solvers.utils import SolverLookup, param_combinations
from ..solvers.solver_interface import ExitStatus


class Probe:
    def geometric_sequence(i):
        return 1.2 * i

    @staticmethod
    def luby_sequence(i, current_timeout, timeout_list):
        timeout_list.append(current_timeout)
        sequence = [timeout_list[0]]
        while len(sequence) <= i:
            sequence += sequence + [2 * sequence[-1]]
        return sequence[i]

    def __init__(self, solvername, model, time_limit, max_tries, all_config, default_config, fix_params, **kwargs):
        self.solvername = solvername
        self.time_limit = time_limit
        self.model = model
        self.global_timeout = time_limit
        self.max_tries = max_tries
        self.all_config = all_config
        self.default_config = default_config
        self.param_order = list(self.all_config.keys())
        self.best_config = self._params_to_np([self.default_config])
        self.probe_timeout = None
        self.round_timeout = None
        self.timeout_list = []
        self.none_change_flag = False
        self.solution_list = []
        self.unchanged_count = 0
        self.fix_params = fix_params
        self.best_params = None
        self.best_runtime = None
        self.default_flag = True
        self.additional_params = kwargs
        print("self.additional_params", self.additional_params)
        self.init_round_type = kwargs.get('init_round_type', None)
        self.stop_type = kwargs.get('stop_type', None)
        self.tuning_timeout_type = kwargs.get('tuning_timeout_type', None)
        self.time_evol = kwargs.get('time_evol', None)
        HPO = kwargs.get('HPO', None)
        self.round_timeout = Probe.initialize_round_timeout(self, self.solvername, self.model, self.init_round_type)
        self.stop = Probe.stop_condition(self, self.stop_type)
        if HPO is None:
            HPO = "Hamming"
        if HPO == "Hamming":
            Probe.Hamming_Distance(self)
        elif HPO == "Bayesian":
            Probe.Bayesian_Optimization(self)
        elif HPO == "Grid":
            Probe.Grid_Search(self)

    def initialize_round_timeout(self, solvername, model, type):
        if type == "Dynamic":
            solver = SolverLookup.get(solvername, model)
            solver.solve(**self.default_config)
            self.base_runtime = solver.status().runtime
            self.round_timeout = self.base_runtime
        elif type == "Static":
            self.round_timeout = 5
        else:
            self.round_timeout = self.time_limit
        return self.round_timeout

    def stop_condition(self, stop_type):
        if stop_type == "First_Solution":
            self.stop = "First_Solution"
        elif stop_type == "Timeout":
            self.stop = "Timeout"
        return self.stop

    def round_timeout_evolution(self, time_evol, current_timeout):
        if time_evol == "Static":
            round_timeout = current_timeout
        elif time_evol == "Geometric":
            round_timeout = Probe.geometric_sequence(current_timeout)
        elif time_evol == "Luby":
            index = len(self.timeout_list)
            round_timeout = self.luby_sequence(index, current_timeout, self.timeout_list)
        return round_timeout

    def Tuning_global_timeout(self, global_timeout, tuning_timeout_type, solution_list, round_counter, probe_timeout):
        if tuning_timeout_type == "Static":
            # probe_timeout = 0.2 * global_timeout
            probe_timeout = global_timeout # This is probing timeout itself
        elif tuning_timeout_type == "Dynamic":
            print("self.unchanged_count", self.unchanged_count)
            if round_counter < self.max_tries:
                if len(self.solution_list) > 1:
                    if self.solution_list[-1].get('objective') == self.solution_list[-2].get('objective'):
                        self.unchanged_count += 1
                    else:
                        self.unchanged_count = 0
                if self.unchanged_count >= 8:
                    self.none_change_flag = True
            else:
                self.none_change_flag = True

        return probe_timeout, self.none_change_flag

    def memory(self):
        print("MEMORY")

    def _get_score(self, combos):
        """
            Return the hamming distance for each remaining configuration to the current best config.
            Lower score means better configuration, so exploit the current best configuration by only allowing small changes.
        """
        return np.count_nonzero(combos != self.best_config, axis=1)

    def _params_to_np(self,combos):
        arr = [[params[key] for key in self.param_order] for params in combos]
        return np.array(arr)

    def _np_to_params(self,arr):
        return {key: val for key, val in zip(self.param_order, arr)}

    def Hamming_Distance(self):
        if self.time_limit is not None:
            start_time = time.time()
        combos = list(param_combinations(self.all_config))
        combos_np = self._params_to_np(combos)
        self.best_runtime = self.round_timeout

        # Ensure random start
        np.random.shuffle(combos_np)

        i = 0
        if self.max_tries is None:
            max_tries = len(combos_np)
        while len(combos_np) and i < self.max_tries:
            # Make new solver
            solver = SolverLookup.get(self.solvername, self.model)
            # Apply scoring to all combos
            scores = self._get_score(combos_np)
            max_idx = np.where(scores == scores.min())[0][0]
            # Get index of optimal combo
            params_np = combos_np[max_idx]
            # Remove optimal combo from combos
            combos_np = np.delete(combos_np, max_idx, axis=0)
            # Convert numpy array back to dictionary
            params_dict = self._np_to_params(params_np)
            # set fixed params
            params_dict.update(self.fix_params)
            timeout = self.best_runtime
            # set timeout depending on time budget
            if self.time_limit is not None:
                timeout = min(timeout, self.time_limit - (time.time() - start_time))
            # run solver
            solver.solve(**params_dict, time_limit=timeout)
            if solver.status().exitstatus == ExitStatus.OPTIMAL and solver.status().runtime < self.best_runtime:
                self.best_runtime = solver.status().runtime
                # update surrogate
                self.best_config = params_np

            if self.time_limit is not None and (time.time() - start_time) >= self.time_limit:
                break
            i += 1

        self.best_params = self._np_to_params(self.best_config)
        self.best_params.update(self.fix_params)
        print(self.best_params , self.best_runtime)
        return self.best_params , self.best_runtime

    def Grid_Search(self):
        if self.time_limit is not None:
            start_time = time.time()
        self.best_runtime = self.round_timeout

        # Get all possible hyperparameter configurations
        combos = list(param_combinations(self.all_config))
        shuffle(combos)  # test in random order

        if self.max_tries is not None:
            combos = combos[:self.max_tries]

        for params_dict in combos:
            # Make new solver
            solver = SolverLookup.get(self.solvername, self.model)
            # set fixed params
            params_dict.update(self.fix_params)
            timeout = self.best_runtime
            # set timeout depending on time budget
            if self.time_limit is not None:
                timeout = min(timeout, self.time_limit - (time.time() - start_time))
            # run solver
            solver.solve(**params_dict, time_limit=timeout)
            if solver.status().exitstatus == ExitStatus.OPTIMAL and solver.status().runtime < self.best_runtime:
                self.best_runtime = solver.status().runtime
                # update surrogate
                self.best_params = params_dict

            if self.time_limit is not None and (time.time() - start_time) >= self.time_limit:
                break

        # print(self.best_params , self.best_runtime)
        return self.best_params , self.best_runtime

    def Bayesian_Optimization(self):
        current_timeout = self.best_runtime = self.round_timeout
        opt = Optimizer(dimensions=dimensions_aslist(self.all_config), base_estimator="GP", acq_func="EI")
        solver = SolverLookup.get(self.solvername, self.model)
        first_non_none_objective = False
        self.mode = "minimize" if self.model.objective_is_min else "maximize"
        best = float('inf') if self.mode == "minimize" else float('-inf')
        round_counter = total_time_used = solve_call_counter = 0
        self.probe_timeout, self.none_change_flag = Probe.Tuning_global_timeout(self, self.global_timeout, self.tuning_timeout_type, self.solution_list, round_counter, self.probe_timeout)
        self.solution_list.append({'params': self.best_params})
        while (self.tuning_timeout_type == "Static" and total_time_used + current_timeout < self.probe_timeout and current_timeout != 0)or(self.tuning_timeout_type == "Dynamic" and round_counter<self.max_tries):
            print("round_counter", round_counter)
            print("current timeout:", current_timeout)
            params = opt.ask()
            parameters = point_asdict(self.all_config, params) if total_time_used != 0 else self.default_config
            # print("parameters:",parameters)
            print("self.solution_list:",self.solution_list)
            seen = False
            for solution in self.solution_list:
                if solution.get('params') == parameters:
                    seen = True
                    print("solution.get('objective')",solution.get('objective'))
                    obj = solution.get('objective')
                    runtime = solution.get('runtime')
                    status = solution.get('status')
                    break
            if seen:
                print("Parameters seen before. Using stored results.")
                # self.solution_list.append(
                #     {'params': dict(parameters), 'objective': obj, 'runtime': runtime, 'status': status})
                total_time_used += 1
            else:
                if self.stop == "Timeout":# "First_Solution" , "Timeout"
                    solver.solve(**parameters, time_limit=current_timeout)
                elif self.stop == "First_Solution":
                    solver.solve(**parameters)
                if solver.objective_value() is not None:
                    print("solve_call_counter is plus 1")
                    solve_call_counter += 1
                    self.first_non_none_objective = True
                if self.mode == "minimize":
                    if (solver.objective_value() is not None and solver.objective_value() < best) or (solver.objective_value() == best and (self.best_runtime is None or solver.status().runtime < self.best_runtime)):
                        best = solver.objective_value()
                        self.best_params = parameters
                        self.best_runtime = solver.status().runtime
                else:
                    if (solver.objective_value() is not None and solver.objective_value() > best) or (solver.objective_value() == best and (
                            self.best_runtime is None or solver.status().runtime < self.best_runtime)):
                        best = solver.objective_value()
                        self.best_params = parameters
                        self.best_runtime = solver.status().runtime
                total_time_used += current_timeout
                obj = solver.objective_value() if solve_call_counter > 0 else None
                if obj is None:
                    obj = best
                    if not first_non_none_objective:
                        current_timeout = Probe.round_timeout_evolution(self, self.time_evol, current_timeout)
                        current_timeout = round(current_timeout, 2)
                        if self.tuning_timeout_type == "Static" and current_timeout > self.probe_timeout - total_time_used:
                            current_timeout = self.probe_timeout - total_time_used
                else:
                    obj = int(obj)
                    first_non_none_objective = True
                self.solution_list.append({
                    'params': dict(parameters),
                    'objective': solver.objective_value(),
                    'runtime': solver.status().runtime,
                    'status': solver.status().exitstatus
                })
                obj = -obj if self.mode == "maximize" else obj
            if self.tuning_timeout_type == "Dynamic":
                probe_timeout, self.none_change_flag = Probe.Tuning_global_timeout(self, self.global_timeout,
                                                                                   self.tuning_timeout_type,
                                                                                   self.solution_list,
                                                                                   round_counter,
                                                                                   self.probe_timeout)

                if self.none_change_flag:
                    print("I am hereeeeeeeeeeeee")
                    total_time_used = self.probe_timeout
                    break
            if self.tuning_timeout_type == "Static" and total_time_used >= self.probe_timeout:
                print("Timeout reached. Exiting.")
                break
            opt.tell(params, obj)
            round_counter += 1
        solve_call_counter += 1
        best_params = point_asdict(self.all_config, opt.Xi[np.argmin(opt.yi)])
        best_params.update(best_params)
        return self.best_params, self.best_runtime

    def get_best_params_and_runtime(self):
        return self.best_params, self.best_runtime
