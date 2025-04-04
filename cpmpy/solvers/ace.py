#!/usr/bin/env python
"""
    Template file for a new solver interface

    Replace <TEMPLATE> by the solver's name, and implement the missing pieces
    The functions are ordered in a way that could be convenient to
    start from the top and continue in that order

    After you are done filling in the template, remove all comments starting with [GUIDELINE]

    WARNING: do not include the python package at the top of the file,
    as CPMpy should also work without this solver installed.
    To ensure that, include it inside supported() and other functions that need it...
"""
import os
import warnings
from datetime import time

import numpy as np
import pkg_resources
from pkg_resources import VersionConflict


from .solver_interface import SolverInterface, SolverStatus, ExitStatus
from ..exceptions import NotSupportedError
from ..expressions.core import Expression, Comparison, Operator, BoolVal
from ..expressions.variables import _BoolVarImpl, NegBoolView, _IntVarImpl, _NumVarImpl
from ..expressions.globalconstraints import DirectConstraint
from ..expressions.globalconstraints import GlobalConstraint
from ..expressions.utils import is_num, is_any_list, is_boolexpr
from ..transformations.get_variables import get_variables
from ..transformations.flatten_model import flatten_constraint, flatten_objective
from ..transformations.normalize import toplevel_list
from ..transformations.decompose_global import decompose_in_tree
from ..transformations.flatten_model import flatten_constraint
from ..transformations.comparison import only_numexpr_equality
from ..transformations.reification import reify_rewrite, only_bv_reifies


"""
    Interface to ace's API

    <some information on the solver>

    Documentation of the solver's own Python API:
    <URL to docs or source code>

    ===============
    List of classes
    ===============

    .. autosummary::
        :nosignatures:

        CPM_ace
"""

class CPM_ace(SolverInterface):
    """
    Interface to ace's API

    Requires that the 'pycsp3' python package is installed:
    $ pip install pycsp3

    See detailed installation instructions at:
    <URL to detailed solver installation instructions, if any>

    Creates the following attributes (see parent constructor for more):
    - ace_model: object, ace's model object
    """

    @staticmethod
    def supported():
        # try to import the package
        try:
            # from pycsp3 import *
            from pycsp3.solvers.ace import Ace as ace
            # optionally enforce a specific version
            from importlib.metadata import version as get_version
            from packaging import version
            pycsp3_version = get_version("pycsp3")
            if version.parse(pycsp3_version) < version.parse("2.2"):
                import warnings
                warnings.warn(f"CPMpy uses features only available from pycsp3 version 2.2, but you have version {pycsp3_version}")
                return False
            pkg_resources.require("pycsp3>=2.2")
            return True
        except ModuleNotFoundError: # if solver's Python package is not installed
            return False
        except VersionConflict: # unsupported version of pycsp3 (optional)
            warnings.warn(f"CPMpy uses features only available from pycsp3 version 2.2, "
                          f"but you have version {pkg_resources.get_distribution('pycsp3').version}.")
            return False
        except Exception as e:
            raise e
        except ImportError:
            return False


    def __init__(self, cpm_model=None, subsolver=None):
        """
        Constructor of the native solver object

        Arguments:
        - cpm_model: Model(), a CPMpy Model() (optional)
        - subsolver: str, name of a subsolver (optional)
        """
        if not self.supported():
            raise Exception("CPM_ace: Install the python package 'pycsp3' to use this solver interface.")

        import pycsp3 as ace
        from pycsp3.solvers.ace import Ace as ace

        assert subsolver is None # unless you support subsolvers, see pysat or minizinc

        # initialise the native solver object
        # [GUIDELINE] we commonly use 3-letter abbrivations to refer to native objects:
        #           OR-tools uses ort_solver, Gurobi grb_solver, Exact xct_solver...
        print(dir(ace))


        self.ace_model = ace.Model()
        self.ace_solver = ace.Solver("ace")

        # for the objective
        self.obj = None
        self.minimize_obj = None
        self.helper_var = None

        # initialise everything else and post the constraints/objective
        # [GUIDELINE] this superclass call should happen AFTER all solver-native objects are created.
        #           internally, the constructor relies on __add__ which uses the above solver native object(s)
        super().__init__(name="ace", cpm_model=cpm_model)


    def solve(self, time_limit=None, **kwargs):
        """
            Call the ace solver

            Arguments:
            - time_limit:  maximum solve time in seconds (float, optional)
            - kwargs:      any keyword argument, sets parameters of solver object

            Arguments that correspond to solver parameters:
            # [GUIDELINE] Please document key solver arguments that the user might wish to change
            #       for example: assumptions=[x,y,z], log_output=True, var_ordering=3, num_cores=8, ...
            # [GUIDELINE] Add link to documentation of all solver parameters
        """

        # ensure all vars are known to solver
        self.solver_vars(list(self.user_vars))
        self.ace_solver = self.ace_model.get_solver()

        start = time.time()
        if time_limit is not None:
            self.ace_solver.set_timelimit_seconds(time_limit)

        # [GUIDELINE] if your solver supports solving under assumptions, add `assumptions` as argument in header
        #       e.g., def solve(self, time_limit=None, assumptions=None, **kwargs):
        #       then translate assumptions here; assumptions are a list of Boolean variables or NegBoolViews

        # call the solver, with parameters
        my_status = self.ace_solver.solve(**kwargs)
        # [GUIDELINE] consider saving the status as self.ace_status so that advanced CPMpy users can access the status object.
        #       This is mainly useful when more elaborate information about the solve-call is saved into the status

        # new status, translate runtime
        self.cpm_status = SolverStatus(self.name)
        self.cpm_status.runtime = self.ace_solver.time() # wallclock time in (float) seconds

        # translate solver exit status to CPMpy exit status
        if my_status is True:
            self.cpm_status.exitstatus = ExitStatus.FEASIBLE
        elif my_status is False:
            self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
        elif my_status is None:
            # can happen when timeout is reached...
            self.cpm_status.exitstatus = ExitStatus.UNKNOWN
        else:  # another?
            raise NotImplementedError(my_status)  # a new status type was introduced, please report on github

        # True/False depending on self.cpm_status
        has_sol = self._solve_return(self.cpm_status)

        # translate solution values (of user specified variables only)
        self.objective_value_ = None
        if has_sol:
            # fill in variable values
            for cpm_var in self.user_vars:
                sol_var = self.solver_var(cpm_var)
                cpm_var._value = self.ace_solver.value(sol_var)
                raise NotImplementedError("ace: back-translating the solution values")

            # translate objective, for optimisation problems only
            if self.has_objective():
                self.objective_value_ = self.ace_solver.ObjectiveValue()

        else: # clear values of variables
            for cpm_var in self.user_vars:
                cpm_var.clear()

        return has_sol


    def solver_var(self, cpm_var):
        """
            Creates solver variable for cpmpy variable
            or returns from cache if previously created
        """
        if is_num(cpm_var): # shortcut, eases posting constraints
            return cpm_var

        # [GUIDELINE] some solver interfaces explicitely create variables on a solver object
        #       then use self.ace_solver.NewBoolVar(...) instead of pycsp3.NewBoolVar(...)

        # special case, negative-bool-view
        # work directly on var inside the view
        if isinstance(cpm_var, NegBoolView):
            return self.ace_model.negate(self.solver_var(cpm_var._bv))

        # create if it does not exist
        if cpm_var not in self._varmap:
            if isinstance(cpm_var, _BoolVarImpl):
                revar = self.ace_model.NewBoolVar(str(cpm_var))
            elif isinstance(cpm_var, _IntVarImpl):
                revar = self.ace_model.NewIntVar(cpm_var.lb, cpm_var.ub, str(cpm_var))
            else:
                raise NotImplementedError("Not a known var {}".format(cpm_var))
            self._varmap[cpm_var] = revar

        # return from cache
        return self._varmap[cpm_var]


    # [GUIDELINE] if ace does not support objective functions, you can delete this function definition
    def objective(self, expr, minimize=True):
        """
            Post the given expression to the solver as objective to minimize/maximize

            'objective()' can be called multiple times, only the last one is stored

            (technical side note: any constraints created during conversion of the objective

            are permanently posted to the solver)
        """
        # make objective function non-nested
        (flat_obj, flat_cons) = flatten_objective(expr)
        self += flat_cons # add potentially created constraints
        self.user_vars.update(get_variables(flat_obj)) # add objvars to vars

        # make objective function or variable and post
        obj = self._make_numexpr(flat_obj)
        # [GUIDELINE] if the solver interface does not provide a solver native "numeric expression" object,
        #         _make_numexpr may be removed and an objective can be posted as:
        #           self.ace_solver.MinimizeWeightedSum(obj.args[0], self.solver_vars(obj.args[1]) or similar

        if minimize:
            self.ace_solver.Minimize(obj)
        else:
            self.ace_solver.Maximize(obj)

    def has_objective(self):
        return self.ace_solver.hasObjective()

    def _make_numexpr(self, cpm_expr):
        """
            Converts a numeric CPMpy 'flat' expression into a solver-specific numeric expression

            Primarily used for setting objective functions, and optionally in constraint posting
        """

        # [GUIDELINE] not all solver interfaces have a native "numerical expression" object.
        #       in that case, this function may be removed and a case-by-case analysis of the numerical expression
        #           used in the constraint at hand is required in __add__
        #       For an example of such solver interface, check out solvers/choco.py or solvers/exact.py

        if is_num(cpm_expr):
            return cpm_expr

        # decision variables, check in varmap
        if isinstance(cpm_expr, _NumVarImpl):  # _BoolVarImpl is subclass of _NumVarImpl
            return self.solver_var(cpm_expr)

        # any solver-native numerical expression
        if isinstance(cpm_expr, Operator):
           if cpm_expr.name == 'sum':
               return self.ace_solver.sum(self.solver_vars(cpm_expr.args))
           elif cpm_expr.name == 'wsum':
               weights, vars = cpm_expr.args
               return self.ace_solver.weighted_sum(weights, self.solver_vars(vars))
           # [GUIDELINE] or more fancy ones such as max
           #        be aware this is not the Maximum CONSTRAINT, but rather the Maximum NUMERICAL EXPRESSION
           elif cpm_expr.name == "max":
               return self.ace_solver.maximum_of_vars(self.solver_vars(cpm_expr.args))
           # ...
        raise NotImplementedError("ace: Not a known supported numexpr {}".format(cpm_expr))


    # `__add__()` first calls `transform()`
    def transform(self, cpm_expr):
        """
            Transform arbitrary CPMpy expressions to constraints the solver supports

            Implemented through chaining multiple solver-independent **transformation functions** from
            the `cpmpy/transformations/` directory.

            See the 'Adding a new solver' docs on readthedocs for more information.

        :param cpm_expr: CPMpy expression, or list thereof
        :type cpm_expr: Expression or list of Expression

        :return: list of Expression
        """
        # apply transformations
        # XXX chose the transformations your solver needs, see cpmpy/transformations/
        cpm_cons = toplevel_list(cpm_expr)
        cpm_cons = decompose_in_tree(cpm_cons, supported={"alldifferent"})
        cpm_cons = flatten_constraint(cpm_cons)  # flat normal form
        cpm_cons = reify_rewrite(cpm_cons, supported=frozenset(['sum', 'wsum']))  # constraints that support reification
        cpm_cons = only_bv_reifies(cpm_cons)
        cpm_cons = only_numexpr_equality(cpm_cons, supported=frozenset(["sum", "wsum", "sub"]))  # supports >, <, !=
        # ...
        return cpm_cons

    def __add__(self, cpm_expr_orig):
        """
            Eagerly add a constraint to the underlying solver.

            Any CPMpy expression given is immediately transformed (through `transform()`)
            and then posted to the solver in this function.

            This can raise 'NotImplementedError' for any constraint not supported after transformation

            The variables used in expressions given to add are stored as 'user variables'. Those are the only ones
            the user knows and cares about (and will be populated with a value after solve). All other variables
            are auxiliary variables created by transformations.

        :param cpm_expr: CPMpy expression, or list thereof
        :type cpm_expr: Expression or list of Expression

        :return: self
        """

        # add new user vars to the set
        get_variables(cpm_expr_orig, collect=self.user_vars)

        # transform and post the constraints
        for cpm_expr in self.transform(cpm_expr_orig):

            if isinstance(cpm_expr, _BoolVarImpl):
                # base case, just var or ~var
                self.ace_solver.add_clause([ self.solver_var(cpm_expr) ])

            elif isinstance(cpm_expr, Operator):
                if cpm_expr.name == "or":
                    self.ace_solver.add_clause(self.solver_vars(cpm_expr.args))
                elif cpm_expr.name == "->": # half-reification
                    bv, subexpr = cpm_expr.args
                    # [GUIDELINE] example code for a half-reified sum/wsum comparison e.g. BV -> sum(IVs) >= 5
                    if isinstance(subexpr, Comparison):
                        lhs, rhs = subexpr.args
                        if isinstance(lhs, _NumVarImpl) or (isinstance(lhs, Operator) and lhs.name in {"sum", "wsum"}):
                            ace_lhs = self._make_numexpr(lhs)
                            self.ace_solver.add_half_reified_comparison(self.solver_var(bv),
                                                                        ace_lhs, subexpr.name, self.solver_var(rhs))
                        else:
                            raise NotImplementedError("ace: no support for half-reified comparison:", subexpr)
                    else:
                        raise NotImplementedError("ace: no support for half-reified constraint:", subexpr)

            elif isinstance(cpm_expr, Comparison):
                lhs, rhs = cpm_expr.args

                # [GUIDELINE] == is used for both double reification and numerical comparisons
                #       need case by case analysis here. Note that if your solver does not support full-reification,
                #       you can rely on the transformation only_implies to convert all reifications to half-reification
                #       for more information, please reach out on github!
                if cpm_expr.name == "==" and is_boolexpr(lhs) and is_boolexpr(rhs): # reification
                    bv, subexpr = lhs, rhs
                    assert isinstance(lhs, _BoolVarImpl), "lhs of reification should be var because of only_bv_reifies"

                    if isinstance(subexpr, Comparison):
                        lhs, rhs = subexpr.args
                        if isinstance(lhs, _NumVarImpl) or (isinstance(lhs, Operator) and lhs.name in {"sum", "wsum"}):
                            ace_lhs = self._make_numexpr(lhs)
                            self.ace_solver.add_reified_comparison(self.solver_var(bv),
                                                                   ace_lhs, subexpr.name, self.solver_var(rhs))
                        else:
                            raise NotImplementedError("ace: no support for reified comparison:", subexpr)
                    else:
                        raise NotImplementedError("ace: no support for reified constraint:", subexpr)

                # otherwise, numerical comparisons
                if isinstance(lhs, _NumVarImpl) or (isinstance(lhs, Operator) and lhs.name in {"sum", "wsum"}):
                    ace_lhs = self._make_numexpr(lhs)
                    self.ace_solver.add_comparison(ace_lhs, cpm_expr.name, self.solver_var(rhs))
                # global functions
                elif cpm_expr.name == "==":
                    ace_rhs = self.solver_var(rhs)
                    if lhs.name == "max":
                        self.ace_solver.add_max_constraint(self.solver_vars(lhs), ace_rhs)
                    elif lhs.name == "element":
                        ace_arr, ace_idx = self.solver_vars(lhs.args)
                        self.ace_solver.add_element_constraint(ace_arr, ace_idx, ace_rhs)
                    # elif...
                    else:
                        raise NotImplementedError("ace: unknown equality constraint:", cpm_expr)
                else:
                    raise NotImplementedError("ace: unknown comparison constraint", cpm_expr)

            # global constraints
            elif cpm_expr.name == "alldifferent":
                self.ace_solver.add_alldifferent(self.solver_vars(cpm_expr.args))
            else:
                raise NotImplementedError("ace: constraint not (yet) supported", cpm_expr)

        return self

    # Other functions from SolverInterface that you can overwrite:
    # solveAll, solution_hint, get_core

    def solveAll(self, display=None, time_limit=None, solution_limit=None, call_from_model=False, **kwargs):
        """
            A shorthand to (efficiently) compute all (optimal) solutions, map them to CPMpy and optionally display the solutions.

            If the problem is an optimization problem, returns only optimal solutions.

           Arguments:
                - display: either a list of CPMpy expressions, OR a callback function, called with the variables after value-mapping
                        default/None: nothing displayed
                - time_limit: stop after this many seconds (default: None)
                - solution_limit: stop after this many solutions (default: None)
                - call_from_model: whether the method is called from a CPMpy Model instance or not
                - any other keyword argument

            Returns: number of solutions found
        """

        # check if objective function
        if self.has_objective():
            raise NotSupportedError("ace does not support finding all optimal solutions")

        # A. Example code if solver supports callbacks
        if is_any_list(display):
            callback = lambda : print([var.value() for var in display])
        else:
            callback = display

        self.solve(time_limit, callback=callback, enumerate_all_solutions=True, **kwargs)
        # clear user vars if no solution found
        if self.ace_solver.SolutionCount() == 0:
            for var in self.user_vars:
                var.clear()
        return self.ace_solver.SolutionCount()

        # B. Example code if solver does not support callbacks
        self.solve(time_limit, enumerate_all_solutions=True, **kwargs)
        solution_count = 0
        for solution in self.aceace_solver.GetAllSolutions():
            solution_count += 1
            # Translate solution to variables
            for cpm_var in self.user_vars:
                cpm_var._value = solution.value(solver_var)

            if display is not None:
                if isinstance(display, Expression):
                    print(display.value())
                elif isinstance(display, list):
                    print([v.value() for v in display])
                else:
                    display()  # callback

        # clear user vars if no solution found
        if solution_count == 0:
            for var in self.user_vars:
                var.clear()

        return solution_count

    def _get_constraint(self, cpm_expr):
        """
        Get a solver's constraint by a supported CPMpy constraint

        :param cpm_expr: CPMpy expression
        :type cpm_expr: Expression

        """

        # Operators: base (bool), lhs=numexpr, lhs|rhs=boolexpr (reified ->)
        if isinstance(cpm_expr, Operator):
            # 'and'/n, 'or'/n, '->'/2
            if cpm_expr.name == 'and':
                return self.ace_model.and_(self.solver_vars(cpm_expr.args))
            elif cpm_expr.name == 'or':
                return self.ace_model.or_(self.solver_vars(cpm_expr.args))

            elif cpm_expr.name == "->":
                cond, subexpr = cpm_expr.args
                if isinstance(cond, _BoolVarImpl) and isinstance(subexpr, _BoolVarImpl):
                    return self.ace_model.or_(self.solver_vars([~cond, subexpr]))
                elif isinstance(cond, _BoolVarImpl):
                    return self._get_constraint(subexpr).implied_by(self.solver_var(cond))
                elif isinstance(subexpr, _BoolVarImpl):
                    return self._get_constraint(cond).implies(self.solver_var(subexpr))
                else:
                    ValueError(f"Unexpected implication: {cpm_expr}")

            else:
                raise NotImplementedError("Not a known supported ACE Operator '{}' {}".format(
                    cpm_expr.name, cpm_expr))

        # Comparisons: both numeric and boolean ones
        # numexpr `comp` bvar|const
        elif isinstance(cpm_expr, Comparison):
            lhs, rhs = cpm_expr.args
            op = "=" if cpm_expr.name == "==" else cpm_expr.name
            if is_boolexpr(lhs) and is_boolexpr(rhs):  # boolean equality -- Reification
                if isinstance(lhs, _BoolVarImpl) and isinstance(lhs, _BoolVarImpl):
                    return self.ace_model.all_equal(self.solver_vars([lhs, rhs]))
                elif isinstance(lhs, _BoolVarImpl):
                    return self._get_constraint(rhs).reify_with(self.solver_var(lhs))
                elif isinstance(rhs, _BoolVarImpl):
                    return self._get_constraint(lhs).reify_with(self.solver_var(rhs))
                else:
                    raise ValueError(f"Unexpected reification {cpm_expr}")

            elif isinstance(lhs, _NumVarImpl):
                return self.ace_model.arithm(self.solver_var(lhs), op, self.solver_var(rhs))
            elif isinstance(lhs, Operator) and lhs.name in {'sum', 'wsum', 'sub'}:
                if lhs.name == 'sum':
                    return self.ace_model.sum(self.solver_vars(lhs.args), op, self.solver_var(rhs))
                elif lhs.name == "sub":
                    a, b = self.solver_vars(lhs.args)
                    return self.ace_model.arithm(a, "-", b, op, self.solver_var(rhs))
                elif lhs.name == 'wsum':
                    wgt, x = lhs.args
                    w = np.array(wgt).tolist()
                    x = self.solver_vars(lhs.args[1])
                    return self.ace_model.scalar(x, w, op, self.solver_var(rhs))

            elif cpm_expr.name == '==':

                chc_rhs = self._to_var(rhs)  # result is always var
                all_vars = {"min", "max", "abs", "div", "mod", "element", "nvalue"}
                if lhs.name in all_vars:

                    chc_args = self._to_vars(lhs.args)

                    if lhs.name == 'min':  # min(vars) = var
                        return self.ace_model.min(chc_rhs, chc_args)
                    elif lhs.name == 'max':  # max(vars) = var
                        return self.ace_model.max(chc_rhs, chc_args)
                    elif lhs.name == 'abs':  # abs(var) = var
                        assert len(chc_args) == 1, f"Expected one argument of abs constraint, but got {chc_args}"
                        return self.ace_model.absolute(chc_rhs, chc_args[0])
                    elif lhs.name == "div":  # var / var = var
                        dividend, divisor = chc_args
                        return self.ace_model.div(dividend, divisor, chc_rhs)
                    elif lhs.name == 'mod':  # var % var = var
                        dividend, divisor = chc_args
                        return self.ace_model.mod(dividend, divisor, chc_rhs)
                    elif lhs.name == "element":  # varsvar[var] = var
                        # TODO: actually, ACE also supports ints[var] = var, but no mix of var and int in array
                        arr, idx = chc_args
                        return self.ace_model.element(chc_rhs, arr, idx)
                    elif lhs.name == "nvalue":  # nvalue(vars) = var
                        # TODO: should look into leaving nvalue <= arg so can post atmost_nvalues here
                        return self.ace_model.n_values(chc_args, chc_rhs)

                elif lhs.name == 'count':  # count(vars, var/int) = var
                    arr, val = lhs.args
                    return self.ace_model.count(self.solver_var(val), self._to_vars(arr), chc_rhs)
                elif lhs.name == "among":
                    arr, vals = lhs.args
                    return self.ace_model.among(chc_rhs, self._to_vars(arr), vals)
                elif lhs.name == 'mul':  # var * var/int = var/int
                    a, b = self.solver_vars(lhs.args)
                    if isinstance(a, int):
                        a, b = b, a  # int arg should always be second
                    return self.ace_model.times(a, b, self.solver_var(rhs))
                elif lhs.name == 'pow':  # var ^ int = var
                    chc_rhs = self._to_var(rhs)
                    return self.ace_model.pow(*self.solver_vars(lhs.args), chc_rhs)

                raise NotImplementedError(
                    "Not a known supported ace left-hand-side '{}' {}".format(lhs.name, cpm_expr))

        # base (Boolean) global constraints
        elif isinstance(cpm_expr, GlobalConstraint):

            # many globals require all variables as arguments
            if cpm_expr.name in {"alldifferent", "alldifferent_except0", "allequal", "circuit", "inverse",
                                 "increasing", "decreasing", "strictly_increasing", "strictly_decreasing",
                                 "lex_lesseq", "lex_less"}:
                chc_args = self._to_vars(cpm_expr.args)
                if cpm_expr.name == 'alldifferent':
                    return self.ace_model.all_different(chc_args)
                elif cpm_expr.name == 'alldifferent_except0':
                    return self.ace_model.all_different_except_0(chc_args)
                elif cpm_expr.name == 'allequal':
                    return self.ace_model.all_equal(chc_args)
                elif cpm_expr.name == "circuit":
                    return self.ace_model.circuit(chc_args)
                elif cpm_expr.name == "inverse":
                    return self.ace_model.inverse_channeling(*chc_args)
                elif cpm_expr.name == "increasing":
                    return self.ace_model.increasing(chc_args, 0)
                elif cpm_expr.name == "decreasing":
                    return self.ace_model.decreasing(chc_args, 0)
                elif cpm_expr.name == "strictly_increasing":
                    return self.ace_model.increasing(chc_args, 1)
                elif cpm_expr.name == "strictly_decreasing":
                    return self.ace_model.decreasing(chc_args, 1)
                elif cpm_expr.name in ["lex_lesseq", "lex_less"]:
                    if cpm_expr.name == "lex_lesseq":
                        return self.ace_model.lex_less_eq(*chc_args)
                    return self.ace_model.lex_less(*chc_args)
            # Ready for when it is fixed in pychoco (https://github.com/chocoteam/pychoco/issues/30)
            #                elif cpm_expr.name == "lex_chain_less":
            #                    return self.ace_model.lex_chain_less(chc_args)

            # but not all
            elif cpm_expr.name == 'table':
                assert (len(cpm_expr.args) == 2)  # args = [array, table]
                array, table = self.solver_vars(cpm_expr.args)
                return self.ace_model.table(array, table)
            elif cpm_expr.name == 'negative_table':
                assert (len(cpm_expr.args) == 2)  # args = [array, table]
                array, table = self.solver_vars(cpm_expr.args)
                return self.ace_model.table(array, table, False)
            elif cpm_expr.name == 'InDomain':
                assert len(cpm_expr.args) == 2  # args = [array, list of vals]
                expr, table = self.solver_vars(cpm_expr.args)
                return self.ace_model.member(expr, table)
            elif cpm_expr.name == "cumulative":
                start, dur, end, demand, cap = cpm_expr.args
                # start, end, demand and cap should be var
                start, end, demand, cap = self._to_vars([start, end, demand, cap])
                # duration can be var or int
                dur = self.solver_vars(dur)
                # Create task variables. ace can create them only one by one
                tasks = [self.ace_model.task(s, d, e) for s, d, e in zip(start, dur, end)]
                return self.ace_model.cumulative(tasks, demand, cap)
            elif cpm_expr.name == "precedence":
                return self.ace_model.int_value_precede_chain(self._to_vars(cpm_expr.args[0]), cpm_expr.args[1])
            elif cpm_expr.name == "gcc":
                vars, vals, occ = cpm_expr.args
                return self.ace_model.global_cardinality(*self.solver_vars([vars, vals]), self._to_vars(occ),
                                                         cpm_expr.closed)
            else:
                raise NotImplementedError(
                    f"Unknown global constraint {cpm_expr}, should be decomposed! If you reach this, please report on github.")

        # unlikely base case: Boolean variable
        elif isinstance(cpm_expr, _BoolVarImpl):
            return self.ace_model.and_([self.solver_var(cpm_expr)])

        # unlikely base case: True or False
        elif isinstance(cpm_expr, BoolVal):
            # ace does not allow to post True or False. Post "certainly True or False" constraints instead
            if cpm_expr.args[0] is True:
                return None
            else:
                if self.helper_var is None:
                    self.helper_var = self.ace_model.intvar(0, 0)
                return self.ace_model.arithm(self.helper_var, "<", 0)

        # a direct constraint, pass to solver
        elif isinstance(cpm_expr, DirectConstraint):
            c = cpm_expr.callSolver(self, self.ace_model)
            return c

        # else
        raise NotImplementedError(cpm_expr)  # if you reach this... please report on github