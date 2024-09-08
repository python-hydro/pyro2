#!/usr/bin/env python3


import argparse
import importlib
import os

import matplotlib.pyplot as plt

import pyro.util.io_pyro as io
import pyro.util.profile_pyro as profile
from pyro.util import compare, msg
from pyro.util.runparams import RuntimeParameters, _get_val

valid_solvers = ["advection",
                 "advection_nonuniform",
                 "advection_rk",
                 "advection_fv4",
                 "advection_weno",
                 "burgers",
                 "viscous_burgers",
                 "compressible",
                 "compressible_rk",
                 "compressible_fv4",
                 "compressible_sdc",
                 "compressible_react",
                 "diffusion",
                 "incompressible",
                 "incompressible_viscous",
                 "lm_atm",
                 "swe"]


class Pyro:
    """
    The main driver to run pyro.
    """

    def __init__(self, solver_name, *, from_commandline=False):
        """
        Constructor

        Parameters
        ----------
        solver_name : str
            Name of solver to use
        from_commandline : bool
            True if we are running from the commandline -- this enables
            runtime vis by default.
        """

        if from_commandline:
            msg.bold('pyro ...')

        if solver_name not in valid_solvers:
            msg.fail(f"ERROR: {solver_name} is not a valid solver")

        self.from_commandline = from_commandline

        self.pyro_home = os.path.dirname(os.path.realpath(__file__)) + '/'
        if not solver_name.startswith("pyro."):
            solver_import = "pyro." + solver_name
        else:
            solver_import = solver_name

        # import desired solver under "solver" namespace
        self.solver = importlib.import_module(solver_import)
        self.solver_name = solver_name

        self.problem_name = None
        self.problem_func = None
        self.problem_params = None
        self.problem_finalize = None

        # custom problems

        self.custom_problems = {}

        # runtime parameters

        # parameter defaults
        self.rp = RuntimeParameters()
        self.rp.load_params(self.pyro_home + "_defaults")
        self.rp.load_params(self.pyro_home + self.solver_name + "/_defaults")

        self.tc = profile.TimerCollection()

        self.is_initialized = False

    def add_problem(self, name, problem_func, *, problem_params=None):
        """Add a problem setup for this solver.

        Parameters
        ----------
        name : str
            The descriptive name of the problem
        problem_func : function
            The function to initialize the state data
        problem_params : dict
            A dictionary of runtime parameters needed for the problem setup
        """

        if problem_params is None:
            problem_params = {}
        self.custom_problems[name] = (problem_func, problem_params)

    def initialize_problem(self, problem_name, *, inputs_file=None, inputs_dict=None):
        """
        Initialize the specific problem

        Parameters
        ----------
        problem_name : str
            Name of the problem
        inputs_file : str
            Filename containing problem's runtime parameters
        inputs_dict : dict
            Dictionary containing extra runtime parameters
        """
        # pylint: disable=attribute-defined-outside-init

        if problem_name in self.custom_problems:
            # this is a problem we added via self.add_problem
            self.problem_name = problem_name
            self.problem_func, self.problem_params = self.custom_problems[problem_name]
            self.problem_finalize = None

        else:
            problem = importlib.import_module("pyro.{}.problems.{}".format(self.solver_name, problem_name))
            self.problem_name = problem_name
            self.problem_func = problem.init_data
            self.problem_params = problem.PROBLEM_PARAMS
            self.problem_finalize = problem.finalize

            if inputs_file is None:
                inputs_file = problem.DEFAULT_INPUTS

        # problem-specific runtime parameters
        for k, v in self.problem_params.items():
            self.rp.set_param(k, v, no_new=False)

        # now read in the inputs file
        if inputs_file is not None:
            if not os.path.isfile(inputs_file):
                # check if the param file lives in the solver's problems directory
                inputs_file = self.pyro_home + self.solver_name + "/problems/" + inputs_file
                if not os.path.isfile(inputs_file):
                    msg.fail("ERROR: inputs file does not exist")

            self.rp.load_params(inputs_file, no_new=1)

        # manually override the I/O, dovis, and verbose defaults
        # for Jupyter, we want runtime vis disabled by default
        if not self.from_commandline:
            self.rp.set_param("vis.dovis", 0)
            self.rp.set_param("driver.verbose", 0)
            self.rp.set_param("io.do_io", 0)

        if inputs_dict is not None:
            for k, v in inputs_dict.items():
                self.rp.set_param(k, v)

        # write out the inputs.auto
        self.rp.print_paramfile()

        self.verbose = self.rp.get_param("driver.verbose")
        self.dovis = self.rp.get_param("vis.dovis")

        # -------------------------------------------------------------------------
        # initialization
        # -------------------------------------------------------------------------

        # initialize the Simulation object -- this will hold the grid and
        # data and know about the runtime parameters and which problem we
        # are running
        self.sim = self.solver.Simulation(
            self.solver_name, self.problem_name, self.problem_func, self.rp,
            problem_finalize_func=self.problem_finalize, timers=self.tc)

        self.sim.initialize()
        self.sim.preevolve()

        plt.ion()

        self.sim.cc_data.t = 0.0

        self.is_initialized = True

    def run_sim(self):
        """
        Evolve entire simulation
        """

        if not self.is_initialized:
            msg.fail("ERROR: problem has not been initialized")

        tm_main = self.tc.timer("main")
        tm_main.begin()

        # output the 0th data
        basename = self.rp.get_param("io.basename")
        do_io = self.rp.get_param("io.do_io")

        if do_io:
            self.sim.write(f"{basename}{self.sim.n:04d}")

        if self.dovis:
            plt.figure(num=1, figsize=(8, 6), dpi=100, facecolor='w')
            self.sim.dovis()

        while not self.sim.finished():
            self.single_step()

        # final output
        force_final_output = self.rp.get_param("io.force_final_output")

        if do_io or force_final_output:
            if self.verbose > 0:
                msg.warning("outputting...")
            basename = self.rp.get_param("io.basename")
            self.sim.write(f"{basename}{self.sim.n:04d}")

        tm_main.end()
        # -------------------------------------------------------------------------
        # final reports
        # -------------------------------------------------------------------------
        if self.verbose > 0:
            self.rp.print_unused_params()
            self.tc.report()

        self.sim.finalize()

    def single_step(self):
        """
        Do a single step
        """

        if not self.is_initialized:
            msg.fail("ERROR: problem has not been initialized")

        # fill boundary conditions
        self.sim.cc_data.fill_BC_all()

        # get the timestep
        self.sim.compute_timestep()

        # evolve for a single timestep
        self.sim.evolve()

        if self.verbose > 0:
            print("%5d %10.5f %10.5f" %
                  (self.sim.n, self.sim.cc_data.t, self.sim.dt))

        # output
        if self.sim.do_output():
            if self.verbose > 0:
                msg.warning("outputting...")
            basename = self.rp.get_param("io.basename")
            self.sim.write(f"{basename}{self.sim.n:04d}")

        # visualization
        if self.dovis:
            tm_vis = self.tc.timer("vis")
            tm_vis.begin()

            self.sim.dovis()
            store = self.rp.get_param("vis.store_images")

            if store == 1:
                basename = self.rp.get_param("io.basename")
                plt.savefig(f"{basename}{self.sim.n:04d}.png")

            tm_vis.end()

    def __repr__(self):
        s = f"Pyro('{self.solver_name}')"
        return s

    def __str__(self):
        s = f"Solver = {self.solver_name}\n"
        if self.is_initialized:
            s += f"Problem = {self.sim.problem_name}\n"
            s += f"Simulation time = {self.sim.cc_data.t}\n"
            s += f"Simulation step number = {self.sim.n}\n"
        s += "\nRuntime Parameters"
        s += "\n------------------\n"
        s += str(self.rp)
        return s

    def get_var(self, v):
        """Alias for the data's get_var routine, returns the
        simulation data given the variable name v.

        """

        if not self.is_initialized:
            msg.fail("ERROR: problem has not been initialized")

        return self.sim.cc_data.get_var(v)

    def get_grid(self):
        """Return the underlying grid object for the simulation

        """

        if not self.is_initialized:
            msg.fail("ERROR: problem has not been initialized")

        return self.sim.cc_data.grid

    def get_sim(self):
        """Return the Simulation object"""
        return self.sim


class PyroBenchmark(Pyro):
    """A subclass of Pyro for benchmarking. Inherits everything from
    pyro, but adds benchmarking routines.

    """

    def __init__(self, solver_name, *,
                 comp_bench=False, reset_bench_on_fail=False, make_bench=False):
        """
        Constructor

        Parameters
        ----------
        solver_name : str
            Name of solver to use
        comp_bench : bool
            Are we comparing to a benchmark?
        reset_bench_on_fail : bool
            Do we reset the benchmark on fail?
        make_bench : bool
            Are we storing a benchmark?
        """

        super().__init__(solver_name)

        self.comp_bench = comp_bench
        self.reset_bench_on_fail = reset_bench_on_fail
        self.make_bench = make_bench

    def run_sim(self, rtol=1.e-12):
        """
        Evolve entire simulation and compare to benchmark at the end.
        """

        super().run_sim()

        result = 0

        if self.comp_bench:
            result = self.compare_to_benchmark(rtol)

        if self.make_bench or (result != 0 and self.reset_bench_on_fail):
            self.store_as_benchmark()

        if self.comp_bench:
            return result
        return self.sim

    def compare_to_benchmark(self, rtol):
        """ Are we comparing to a benchmark? """

        basename = self.rp.get_param("io.basename")
        compare_file = "{}{}/tests/{}{:04d}".format(
            self.pyro_home, self.solver_name, basename, self.sim.n)
        msg.warning(f"comparing to: {compare_file} ")
        try:
            sim_bench = io.read(compare_file)
        except OSError:
            msg.warning("ERROR opening compare file")
            return "ERROR opening compare file"

        result = compare.compare(self.sim.cc_data, sim_bench.cc_data, rtol)

        if result == 0:
            msg.success(f"results match benchmark to within relative tolerance of {rtol}\n")
        else:
            msg.warning("ERROR: " + compare.errors[result] + "\n")

        return result

    def store_as_benchmark(self):
        """ Are we storing a benchmark? """

        if not os.path.isdir(self.pyro_home + self.solver_name + "/tests/"):
            try:
                os.mkdir(self.pyro_home + self.solver_name + "/tests/")
            except (FileNotFoundError, PermissionError):
                msg.fail(
                    "ERROR: unable to create the solver's tests/ directory")

        basename = self.rp.get_param("io.basename")
        bench_file = self.pyro_home + self.solver_name + "/tests/" + \
            basename + "%4.4d" % (self.sim.n)
        msg.warning(f"storing new benchmark: {bench_file}\n")
        self.sim.write(bench_file)


def parse_args():
    """Parse the runtime parameters"""

    p = argparse.ArgumentParser()

    p.add_argument("--make_benchmark",
                   help="create a new benchmark file for regression testing",
                   action="store_true")
    p.add_argument("--compare_benchmark",
                   help="compare the end result to the stored benchmark",
                   action="store_true")

    p.add_argument("solver", metavar="solver-name", type=str, nargs=1,
                   help="name of the solver to use", choices=valid_solvers)
    p.add_argument("problem", metavar="problem-name", type=str, nargs=1,
                   help="name of the problem to run")
    p.add_argument("param", metavar="inputs-file", type=str, nargs=1,
                   help="name of the inputs file")

    p.add_argument("other", metavar="runtime-parameters", type=str, nargs="*",
                   help="additional runtime parameters that override the inputs file "
                   "in the format section.option=value")

    return p.parse_args()


def main():
    args = parse_args()

    if args.compare_benchmark or args.make_benchmark:
        pyro = PyroBenchmark(args.solver[0],
                             comp_bench=args.compare_benchmark,
                             make_bench=args.make_benchmark)
    else:
        pyro = Pyro(args.solver[0], from_commandline=True)

    other = {}
    for param_string in args.other:
        k, v = param_string.split("=")
        other[k] = _get_val(v)

    print(other)
    pyro.initialize_problem(problem_name=args.problem[0],
                            inputs_file=args.param[0],
                            inputs_dict=other)
    pyro.run_sim()


if __name__ == "__main__":
    main()
