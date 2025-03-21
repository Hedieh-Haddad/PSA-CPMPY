import subprocess
import sys
import os
import time
from random import shuffle
import pandas as pd
from skopt import Optimizer
from skopt.utils import point_asdict, dimensions_aslist
import numpy as np
from ..solvers.utils import SolverLookup, param_combinations
from ..solvers.solver_interface import ExitStatus
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

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
        self.mode = "minimize" if self.model.objective_is_min else "maximize"
        print(os.path.basename(sys.argv[0]))
        self.problem_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
        self.script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        self.relative_path = os.path.join(self.script_dir, os.path.basename(sys.argv[0]))
        print(len(sys.argv))
        print(sys.argv[0])
        print(sys.argv[1])
        print(sys.argv[2])
        if len(sys.argv) >= 2 :
            self.HPO = sys.argv[1]
            self.solver_name = sys.argv[2]
        else:
            self.HPO = "Hamming"
            self.solver_name = "ortools"
        self.global_timeout = time_limit
        self.max_tries = max_tries
        self.all_config = all_config
        self.default_config = default_config
        self.param_order = list(self.all_config.keys())
        self.best_config = self._params_to_np([self.default_config])
        self.best_config_str = self.dict_to_string(self.default_config)
        # print("self.best_config_str", self.best_config_str)
        self.probe_timeout = None
        self.solving_time = None
        self.round_timeout = None
        # self.best_obj = None
        self.best_obj = 1e10 if self.mode == "minimize" else -1e10
        self.timeout_list = []
        self.none_change_flag = False
        self.solution_list = []
        self.unchanged_count = 0
        self.fix_params = fix_params
        self.best_params = None
        self.best_runtime = None
        self.default_flag = True
        self.additional_params = kwargs
        print("self.additional_params", self.additional_params, self.HPO)
        self.init_round_type = kwargs.get('init_round_type', None)
        self.stop_type = kwargs.get('stop_type', None)
        self.tuning_timeout_type = kwargs.get('tuning_timeout_type', None)
        self.time_evol = kwargs.get('time_evol', None)
        # self.HPO = kwargs.get('HPO', None)
        self.results_file = "hpo_results.csv"
        self.round_timeout = Probe.initialize_round_timeout(self, self.solvername, self.model, self.init_round_type)
        self.stop = Probe.stop_condition(self, self.stop_type)
        # if self.HPO is None:
        #     self.HPO = "Hamming"
        if self.HPO == "Hamming":
            Probe.Hamming_Distance(self)
        elif self.HPO == "Bayesian":
            Probe.Bayesian_Optimization(self)
        elif self.HPO == "Grid":
            Probe.Grid_Search(self)
        Probe.save_result(self)


    def set_hp(self):
        if self.solver_name == "ortools":
            tunables = {
                'optimize_with_core': [False, True],
                'search_branching': [0, 1, 2, 3, 4, 5, 6],
                'boolean_encoding_level': [0, 1, 2, 3],
                'linearization_level': [0, 1, 2],
                'core_minimization_level': [0, 1, 2],  # new in OR-tools>= v9.8
                'cp_model_probing_level': [0, 1, 2, 3],
                'cp_model_presolve': [False, True],
                'clause_cleanup_ordering': [0, 1],
                'binary_minimization_algorithm': [0, 1, 2, 3, 4],
                'minimization_algorithm': [0, 1, 2, 3],
                'use_phase_saving': [False, True]
                }

            defaults = {
                'optimize_with_core': False,
                'search_branching': 0,
                'boolean_encoding_level': 1,
                'linearization_level': 1,
                'core_minimization_level': 2,# new in OR-tools>=v9.8
                'cp_model_probing_level': 2,
                'cp_model_presolve': True,
                'clause_cleanup_ordering': 0,
                'binary_minimization_algorithm': 1,
                'minimization_algorithm': 2,
                'use_phase_saving': True
                }
        elif self.solver_name == "choco":
            tunables = {
                "solution_limit": [None, 100, 500, 1000],
                "node_limit": [None, 1000, 5000, 10000],
                "fail_limit": [None, 100, 500, 1000],
                "restart_limit": [None, 10, 50, 100],
                "backtrack_limit": [None, 100, 500, 1000]
            }

            defaults = {
                "solution_limit": None,
                "node_limit": None,
                "fail_limit": None,
                "restart_limit": None,
                "backtrack_limit": None
            }


    def initialize_round_timeout(self, solvername, model, type):
        if self.HPO == "Hamming":
            type = "Dynamic"
        if type == "Dynamic":
            if solvername == "ortools":
                solver = SolverLookup.get(solvername, model)
                solver.solve(**self.default_config)
            elif solvername == "choco":
                print("self.relative_path", self.relative_path)
                print("**self.best_config_str", self.best_config_str)
                solver = SolverLookup.get(solvername, model)
                solver.solve(**self.default_config)
            #     # cmd = ["java", "-jar", f"/Users/hedieh.haddad/Desktop/GITHUB/PSA-CPMPY/cpmpy/solvers/choco.jar", self.relative_path, "-f","-csv"]
            #     cmd = ["python", self.relative_path,"-f","-csv"]
            #     # print("model", model)
            #     # cmd = ["python", self.relative_path]
            #     # # cmd.append(f"-valsel=[{valh},true,16,true]")
            #     # # cmd.append(f"-varsel=[{varh},tie,32]")
            #     # cmd.append("-lc=1")
            #     # cmd.append(f"-restarts=[luby,500,50000,true]")
            #     # if varh == "Solver_Default" and valh == "Solver_Default":
            #     #     cmd = ["java", "-jar", f"../../choco-solver-4.10.14/choco-solver/choco.jar", config.model, "-f",
            #     #            "-csv", f"-limit {probe_time}"]
            #     #     cmd.append("-lc=1")
            #     # cmd.append(f"-restarts=[luby,500,50000,true]")
            #     output = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            #     print(output.stdout)
            # output.kill()
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
        elif time_evol == "Dynamic_Geometric":
            round_timeout = Probe.geometric_sequence(current_timeout)
        elif time_evol == "Dynamic_Luby":
            index = len(self.timeout_list)
            round_timeout = self.luby_sequence(index, current_timeout, self.timeout_list)
        return round_timeout

    def Tuning_global_timeout(self, global_timeout, tuning_timeout_type, solution_list, round_counter, probe_timeout, total_time_used):
        if tuning_timeout_type == "Static":
            probe_timeout = 0.2 * global_timeout
            self.solving_time = global_timeout - probe_timeout
            # probe_timeout = global_timeout # This is probing timeout itself
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
                    self.solving_time = global_timeout - total_time_used
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

    def dict_to_string(self, input_dict):
        return ' '.join([f"{key}={value}" for key, value in input_dict.items()])

    def Hamming_Distance(self):
        if self.time_limit is not None:
            start_time = time.time()
        combos = list(param_combinations(self.all_config))
        combos_np = self._params_to_np(combos)
        self.best_runtime = self.round_timeout
        # Ensure random start
        np.random.shuffle(combos_np)
        total_time_used = 0
        i = 0
        if self.max_tries is None:
            self.max_tries = len(combos_np)
        while len(combos_np) and i < self.max_tries and total_time_used < self.global_timeout:
            # Make new solver total_time_used += current_timeout
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
                self.best_obj = solver.objective_value()
            if solver.objective_value() is not None and  solver.objective_value() < self.best_obj:
                self.best_obj = solver.objective_value()
            print("obj",solver.objective_value())
            print("runtime",solver.status().runtime)
            current_timeout = solver.status().runtime
            if self.time_limit is not None and (time.time() - start_time) >= self.time_limit:
                break
            i += 1
            total_time_used += current_timeout

        self.best_params = self._np_to_params(self.best_config)
        self.best_params.update(self.fix_params)
        print(self.best_params , self.best_runtime , self.best_obj)
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
        self.best_obj = obj = 1e10 if self.mode == "minimize" else -1e10
        round_counter = total_time_used = solve_call_counter = seen_counter = 0
        self.probe_timeout, self.none_change_flag = Probe.Tuning_global_timeout(self, self.global_timeout, self.tuning_timeout_type, self.solution_list, round_counter, self.probe_timeout, total_time_used)
        self.solution_list.append({'params': self.best_params})
        while (self.tuning_timeout_type == "Static" and total_time_used + current_timeout < self.probe_timeout and current_timeout != 0)or(self.tuning_timeout_type == "Dynamic" and round_counter<self.max_tries):
            params = opt.ask()
            parameters = point_asdict(self.all_config, params) if total_time_used != 0 else self.default_config
            # print("*************", parameters)
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
                seen_counter += 1
                if obj is None or np.isnan(obj) or np.isinf(obj):
                    if self.mode == 'maximize':
                        obj = -1e10
                    elif self.mode == 'minimize':
                        obj = 1e10
                total_time_used += 1
                # if seen_counter == 8:
                #     break
            else:
                if self.stop == "Timeout":# "First_Solution" , "Timeout"
                    print("**parameters", parameters)
                    # solver.solve(**parameters, time_limit=current_timeout)
                    solver.solve(**parameters, time_limit=current_timeout)
                elif self.stop == "First_Solution":
                    solver.solve(**parameters)
                if solver.objective_value() is not None:
                    solve_call_counter += 1
                    self.first_non_none_objective = True
                if self.mode == "minimize":
                    if (solver.objective_value() is not None and solver.objective_value() < self.best_obj) or (solver.objective_value() == self.best_obj and (self.best_runtime is None or solver.status().runtime < self.best_runtime)):
                        self.best_obj = solver.objective_value()
                        self.best_params = parameters
                        self.best_runtime = solver.status().runtime
                else:
                    if (solver.objective_value() is not None and solver.objective_value() > self.best_obj) or (solver.objective_value() == self.best_obj and (
                            self.best_runtime is None or solver.status().runtime < self.best_runtime)):
                        self.best_obj = solver.objective_value()
                        self.best_params = parameters
                        self.best_runtime = solver.status().runtime
                total_time_used += current_timeout
                obj = solver.objective_value() if solve_call_counter > 0 else None
                if obj is None or not np.isfinite(obj):
                    obj = self.best_obj if self.best_obj is not None and np.isfinite(self.best_obj) else 1e10
                    obj = round(float(obj), 3)
                    if not first_non_none_objective:
                        current_timeout = Probe.round_timeout_evolution(self, self.time_evol, current_timeout)
                        current_timeout = round(current_timeout, 2)
                        if self.tuning_timeout_type == "Static" and current_timeout > self.probe_timeout - total_time_used:
                            current_timeout = self.probe_timeout - total_time_used
                else:
                    obj = round(float(obj), 3)
                    first_non_none_objective = True
                self.solution_list.append({
                    'params': dict(parameters),
                    # 'objective': solver.objective_value(),
                    'objective': obj,
                    'runtime': round(solver.status().runtime, 3),
                    'status': solver.status().exitstatus
                })
                obj = -obj if self.mode == "maximize" else obj
            if self.tuning_timeout_type == "Dynamic":
                probe_timeout, self.none_change_flag = Probe.Tuning_global_timeout(self, self.global_timeout,
                                                                                   self.tuning_timeout_type,
                                                                                   self.solution_list,
                                                                                   round_counter,
                                                                                   self.probe_timeout, total_time_used)
                print("self.none_change_flag", self.none_change_flag)
                if self.none_change_flag:
                    break
            if self.tuning_timeout_type == "Static" and total_time_used >= self.probe_timeout:
                print("Timeout reached. Exiting.")
                break
            opt.tell(params, obj)
            round_counter += 1
        solve_call_counter += 1
        best_params = point_asdict(self.all_config, opt.Xi[np.argmin(opt.yi)])
        best_params.update(best_params)
        solver.solve(**self.best_params, time_limit=self.solving_time)
        print(solver.objective_value())
        print(self.best_params)
        print(solver.status().runtime)
        print(self.best_params , self.best_runtime , self.best_obj)
        return self.best_params, self.best_runtime

    def get_best_params_and_runtime(self):
        return self.best_params, self.best_runtime


    def save_result(self):
        if os.path.exists(self.results_file):
            df = pd.read_csv(self.results_file)
        else:
            df = pd.DataFrame(columns=[
                "problem", "Global_timeout", "Mode",
                "objective_Hamming", "run_time_Hamming", "best_configuration_Hamming",
                "objective_BO", "run_time_BO", "best_configuration_BO",
                "objective_Grid", "run_time_Grid", "best_configuration_Grid",
                "Best_HPO_Method"
            ])

        if self.problem_name in df["problem"].values:
            row_index = df[df["problem"] == self.problem_name].index[0]
        else:
            new_row = {
                "problem": self.problem_name,
                "Global_timeout": self.global_timeout,
                "Mode": self.mode,
                "objective_Hamming": None, "run_time_Hamming": None, "best_configuration_Hamming": None,
                "objective_BO": None, "run_time_BO": None, "best_configuration_BO": None,
                "objective_Grid": None, "run_time_Grid": None, "best_configuration_Grid": None,
                "Best_HPO_Method": None
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            row_index = df.index[-1]

        # Update results
        if self.HPO == "Hamming":
            df.at[row_index, "objective_Hamming"] = self.best_obj
            df.at[row_index, "run_time_Hamming"] = str(self.best_runtime)
            df.at[row_index, "best_configuration_Hamming"] = str(self.best_params)

        elif self.HPO == "Bayesian":
            df.at[row_index, "objective_BO"] = self.best_obj
            df.at[row_index, "run_time_BO"] = str(self.best_runtime)
            df.at[row_index, "best_configuration_BO"] = str(self.best_params)

        elif self.HPO == "Grid":
            df.at[row_index, "objective_Grid"] = self.best_obj
            df.at[row_index, "run_time_Grid"] = str(self.best_runtime)
            df.at[row_index, "best_configuration_Grid"] = str(self.best_params)

        # Compare methods and update best one
        self.compare_hpo_methods(df)

        df.to_csv(self.results_file, index=False)
        print(f"Results updated for {self.problem_name} in {self.results_file}")

    def compare_hpo_methods(self, df):
        for i, row in df.iterrows():
            mode = str(row["Mode"]).strip().lower()

            if mode == "maximize":
                hamming_obj = row["objective_Hamming"] if pd.notna(row["objective_Hamming"]) else -1e10
                bo_obj = row["objective_BO"] if pd.notna(row["objective_BO"]) else -1e10
                grid_obj = row["objective_Grid"] if pd.notna(row["objective_Grid"]) else -1e10
            elif mode == "minimize":
                hamming_obj = row["objective_Hamming"] if pd.notna(row["objective_Hamming"]) else 1e10
                bo_obj = row["objective_BO"] if pd.notna(row["objective_BO"]) else 1e10
                grid_obj = row["objective_Grid"] if pd.notna(row["objective_Grid"]) else 1e10
            else:
                raise ValueError(f"Unknown mode: {mode}")

            print(f"Row {i} - Hamming: {hamming_obj}, BO: {bo_obj}, Grid: {grid_obj}")
            print(f"Mode: {mode}")

            # Convert runtime values to float (set inf if missing)
            hamming_time = float(row["run_time_Hamming"]) if pd.notna(row["run_time_Hamming"]) else float("inf")
            bo_time = float(row["run_time_BO"]) if pd.notna(row["run_time_BO"]) else float("inf")
            grid_time = float(row["run_time_Grid"]) if pd.notna(row["run_time_Grid"]) else float("inf")

            # Determine best HPO method
            best_obj = max(hamming_obj, bo_obj, grid_obj) if mode == "maximize" else min(hamming_obj, bo_obj, grid_obj)
            print(f"Best objective found: {best_obj}")

            # Select the best method (if there's a tie, pick the one with the shortest runtime)
            best_methods = []
            if abs(hamming_obj - best_obj) < 1e-6:
                best_methods.append(("Hamming", hamming_time))
            if abs(bo_obj - best_obj) < 1e-6:
                best_methods.append(("Bayesian", bo_time))
            if abs(grid_obj - best_obj) < 1e-6:
                best_methods.append(("Grid", grid_time))

            print(f"Candidate best methods before sorting: {best_methods}")
            best_methods.sort(key=lambda x: x[1])  # Sort by runtime (ascending)
            df.at[i, "Best_HPO_Method"] = best_methods[0][0]  # Pick the fastest one
            print(f"Selected Best HPO Method: {df.at[i, 'Best_HPO_Method']}")

        df.to_csv(self.results_file, index=False)


    def config(self):
        if self.solvername == "choco":
            tunables = {
                "-a": [False, True],
                "-f": [False, True],
                "-last": [False, True]
            }

            defaults = {
                "-a": False,
                "-f": False,
                "-last": False
            }

        default_params = {
            "init_round_type": "Static",
            "stop_type": "Timeout",
            "tuning_timeout_type": "Static",
            "time_evol": "Dynamic_Geometric",
        }
        user_params = {
            "init_round_type": "Dynamic",  # "Dynamic", "Static" , "None"
            "stop_type": "Timeout",  # "First_Solution" , "Timeout"
            "tuning_timeout_type": "Static",  # "Static" , "Dynamic", "None"
            "time_evol": "Static"  # "Static", "Dynamic_Geometric" , "Dynamic_Luby"
        }
        params = {**default_params, **user_params}