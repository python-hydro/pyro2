#!/usr/bin/env python3

from __future__ import print_function

import argparse
import importlib
import os

import matplotlib.pyplot as plt

import compare
from util import msg, profile, runparams, io


class Pyro(object):
    """
    The main driver to run pyro.

    Notes / TODO:
    * Further decouple some of the benchmarking stuff?
    * Should it be possible to pass in a problem function and initialise a Pyro/
      Simulation object using that? That would also require some modifications
      of the Simulation class, as we currently pass the problem name to its
      constructor. I think this would make sense though if we were moving
      towards something that worked better as a Jupyter Notebook.
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
        msg.bold('pyro ...')

        # import desired solver under "solver" namespace
        self.solver = importlib.import_module(solver_name)
        self.solver_name = solver_name
        self.comp_bench = comp_bench
        self.reset_bench_on_fail = reset_bench_on_fail
        self.make_bench = make_bench

        #-------------------------------------------------------------------------
        # runtime parameters
        #-------------------------------------------------------------------------

        # parameter defaults
        self.rp = runparams.RuntimeParameters()
        self.rp.load_params("_defaults")
        self.rp.load_params(solver_name + "/_defaults")

        self.tc = profile.TimerCollection()

    def initialize_problem(self, problem_name, param_file=None, param_dict=None,
                           other_commands=None):
        """
        Initialize the specific problem

        Parameters
        ----------
        problem_name : str
            Name of the problem
        param_file : str
            Filename containing problem's runtime parameters
        param_dict : dict
            Dictionary containing extra runtime parameters
        other_commands : str
            Other command line parameter options
        """

        # problem-specific runtime parameters
        self.rp.load_params(self.solver_name + "/problems/_" + problem_name + ".defaults")

        # now read in the inputs file
        if param_dict is not None:
            for k, v in param_dict.items():
                self.rp.params[k] = v
        if param_file is not None:
            if not os.path.isfile(param_file):
                # check if the param file lives in the solver's problems directory
                param_file = self.solver_name + "/problems/" + param_file
                if not os.path.isfile(param_file):
                    msg.fail("ERROR: inputs file does not exist")

            self.rp.load_params(param_file, no_new=1)

        # and any commandline overrides
        if other_commands is not None:
            self.rp.command_line_params(other_commands)

        # write out the inputs.auto
        self.rp.print_paramfile()

        self.verbose = self.rp.get_param("driver.verbose")
        self.dovis = self.rp.get_param("vis.dovis")

        #-------------------------------------------------------------------------
        # initialization
        #-------------------------------------------------------------------------

        # initialize the Simulation object -- this will hold the grid and
        # data and know about the runtime parameters and which problem we
        # are running
        self.sim = self.solver.Simulation(self.solver_name, problem_name, self.rp, timers=self.tc)

        self.sim.initialize()
        self.sim.preevolve()

    def run_sim(self):
        """
        Evolve entire simulation
        """

        tm_main = self.tc.timer("main")
        tm_main.begin()

        plt.ion()

        self.sim.cc_data.t = 0.0

        # output the 0th data
        basename = self.rp.get_param("io.basename")
        self.sim.write("{}{:04d}".format(basename, self.sim.n))

        if self.dovis:
            plt.figure(num=1, figsize=(8, 6), dpi=100, facecolor='w')
            self.sim.dovis()

        while not self.sim.finished():

            self.single_step()

        # final output
        if self.verbose > 0:
            msg.warning("outputting...")
        basename = self.rp.get_param("io.basename")
        self.sim.write("{}{:04d}".format(basename, self.sim.n))

        tm_main.end()

        result = self.compare_to_benchmark()
        self.make_bench(result)

        #-------------------------------------------------------------------------
        # final reports
        #-------------------------------------------------------------------------
        if self.verbose > 0:
            self.rp.print_unused_params()
            self.tc.report()

        self.sim.finalize()

        if self.comp_bench:
            return result
        else:
            return self.sim

    def single_step(self):
        """
        Do a single step
        """
        # fill boundary conditions
        self.sim.cc_data.fill_BC_all()

        # get the timestep
        self.sim.compute_timestep()

        # evolve for a single timestep
        self.sim.evolve()

        if self.verbose > 0:
            print("%5d %10.5f %10.5f" % (self.sim.n, self.sim.cc_data.t, self.sim.dt))

        # output
        if self.sim.do_output():
            if self.verbose > 0:
                msg.warning("outputting...")
            basename = self.rp.get_param("io.basename")
            self.sim.write("{}{:04d}".format(basename, self.sim.n))

        # visualization
        if self.dovis:
            tm_vis = self.tc.timer("vis")
            tm_vis.begin()

            self.sim.dovis()
            store = self.rp.get_param("vis.store_images")

            if store == 1:
                basename = self.rp.get_param("io.basename")
                plt.savefig("{}{:04d}.png".format(basename, self.sim.n))

            tm_vis.end()

    def compare_to_benchmark(self):
        """ Are we comparing to a benchmark? """

        result = 0

        if self.comp_bench:
            basename = self.rp.get_param("io.basename")
            compare_file = "{}/tests/{}{:04d}".format(
                self.solver_name, basename, self.sim.n)
            msg.warning("comparing to: {} ".format(compare_file))
            try:
                sim_bench = io.read(compare_file)
            except IOError:
                msg.warning("ERROR openning compare file")
                return "ERROR openning compare file"

            result = compare.compare(self.sim.cc_data, sim_bench.cc_data)

            if result == 0:
                msg.success("results match benchmark\n")
            else:
                msg.warning("ERROR: " + compare.errors[result] + "\n")

        return result

    def store_as_benchmark(self, result):
        """ Are we storing a benchmark? """
        if self.make_bench or (result != 0 and self.reset_bench_on_fail):
            if not os.path.isdir(self.solver_name + "/tests/"):
                try:
                    os.mkdir(self.solver_name + "/tests/")
                except (FileNotFoundError, PermissionError):
                    msg.fail("ERROR: unable to create the solver's tests/ directory")

            basename = self.rp.get_param("io.basename")
            bench_file = self.solver_name + "/tests/" + basename + "%4.4d" % (self.sim.n)
            msg.warning("storing new benchmark: {}\n".format(bench_file))
            self.sim.write(bench_file)


def parse_args():
    """Parse the runtime parameters"""

    valid_solvers = ["advection",
                     "advection_rk",
                     "advection_fv4",
                     "advection_weno",
                     "compressible",
                     "compressible_rk",
                     "compressible_fv4",
                     "compressible_sdc",
                     "compressible_react",
                     "diffusion",
                     "incompressible",
                     "lm_atm",
                     "swe"]

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


if __name__ == "__main__":
    args = parse_args()
    pyro = Pyro(args.solver[0], comp_bench=args.compare_benchmark,
                make_bench=args.make_benchmark)
    pyro.initialize_problem(problem_name=args.problem[0], param_file=args.param[0],
                other_commands=args.other)
    pyro.run_sim()
