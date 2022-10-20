#!/usr/bin/env python3


import argparse
import importlib
import os

import matplotlib.pyplot as plt

import pyro.util.io_pyro as io
import pyro.util.profile_pyro as profile
from pyro.util import compare, msg, runparams

valid_solvers = ["advection",
                 "advection_nonuniform",
                 "advection_rk",
                 "advection_fv4",
                 "advection_weno",
                 "compressible",
                 "compressible_rk",
                 "compressible_fv4",
                 "compressible_sdc",
                 "compressible_react",
                 "compressible_sr",
                 "diffusion",
                 "incompressible",
                 "lm_atm",
                 "swe"]


class Pyro:
    """
    The main driver to run pyro.
    """

    def __init__(self, solver_name):
        """
        Constructor

        Parameters
        ----------
        solver_name : str
            Name of solver to use
        """

        msg.bold('pyro ...')

        if solver_name not in valid_solvers:
            msg.fail(f"ERROR: {solver_name} is not a valid solver")

        self.pyro_home = os.path.dirname(os.path.realpath(__file__)) + '/'
        if not solver_name.startswith("pyro."):
            solver_import = "pyro." + solver_name
        else:
            solver_import = solver_name

        # import desired solver under "solver" namespace
        self.solver = importlib.import_module(solver_import)
        self.solver_name = solver_name

        # -------------------------------------------------------------------------
        # runtime parameters
        # -------------------------------------------------------------------------

        # parameter defaults
        self.rp = runparams.RuntimeParameters()
        self.rp.load_params(self.pyro_home + "_defaults")
        self.rp.load_params(self.pyro_home + self.solver_name + "/_defaults")

        self.tc = profile.TimerCollection()

        self.is_initialized = False

    def initialize_problem(self, problem_name, inputs_file=None, inputs_dict=None,
                           other_commands=None):
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
        other_commands : str
            Other command line parameter options
        """

        problem_defaults_file = self.pyro_home + self.solver_name + \
            "/problems/_" + problem_name + ".defaults"

        # problem-specific runtime parameters
        if os.path.isfile(problem_defaults_file):
            self.rp.load_params(problem_defaults_file)

        # now read in the inputs file
        if inputs_file is not None:
            if not os.path.isfile(inputs_file):
                # check if the param file lives in the solver's problems directory
                inputs_file = self.pyro_home + self.solver_name + "/problems/" + inputs_file
                if not os.path.isfile(inputs_file):
                    msg.fail("ERROR: inputs file does not exist")

            self.rp.load_params(inputs_file, no_new=1)

        if inputs_dict is not None:
            for k, v in inputs_dict.items():
                self.rp.params[k] = v

        # and any commandline overrides
        if other_commands is not None:
            self.rp.command_line_params(other_commands)

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
            self.solver_name, problem_name, self.rp, timers=self.tc)

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
        self.sim.write(f"{basename}{self.sim.n:04d}")

        if self.dovis:
            plt.figure(num=1, figsize=(8, 6), dpi=100, facecolor='w')
            self.sim.dovis()

        while not self.sim.finished():
            self.single_step()

        # final output
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

        return self.sim

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
        """ Return a representation of the Pyro object """
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
        """
        Alias for cc_data's get_var routine, returns the cell-centered data
        given the variable name v.
        """

        if not self.is_initialized:
            msg.fail("ERROR: problem has not been initialized")

        return self.sim.cc_data.get_var(v)


class PyroBenchmark(Pyro):
    """
    A subclass of Pyro for benchmarking. Inherits everything from pyro, but adds benchmarking routines.
    """

    def __init__(self, solver_name, comp_bench=False,
                 reset_bench_on_fail=False, make_bench=False):
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
        else:
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
        pyro = Pyro(args.solver[0])

    pyro.initialize_problem(problem_name=args.problem[0],
                            inputs_file=args.param[0],
                            other_commands=args.other)
    pyro.run_sim()


if __name__ == "__main__":
    main()
