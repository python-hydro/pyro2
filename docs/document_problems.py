#!/usr/bin/env python

import importlib
from pathlib import Path

SOLVERS = ["advection",
           "advection_fv4",
           "advection_nonuniform",
           "advection_rk",
           "advection_weno",
           "burgers",
           "burgers_viscous",
           "compressible",
           "compressible_fv4",
           "compressible_react",
           "compressible_rk",
           "compressible_sdc",
           "diffusion",
           "incompressible",
           "incompressible_viscous",
           "lm_atm",
           "swe"]

MAX_LEN = 36


def doit(pyro_home):

    for s in SOLVERS:

        # check it the problems/ directory is not a softlink to
        # a different solver

        p = Path(f"{pyro_home}/pyro/{s}/problems").resolve()

        with open(f"{pyro_home}/docs/source/{s}_problems.inc", "w") as finc:

            finc.write("supported problems\n")
            finc.write("------------------\n")

            if (parent_solver := p.parts[-2]) == s:

                # find all the problems
                for prob in p.glob("*.py"):
                    if prob.name == "__init__.py":
                        continue

                    mprob = importlib.import_module(f"pyro.{s}.problems.{prob.stem}")

                    if "init_data" not in dir(mprob):
                        # not a problem setup
                        continue

                    finc.write(f"``{prob.name}``\n")
                    finc.write("^" * (len(prob.name)+4) + "\n\n")

                    if mprob.__doc__:
                        finc.write(mprob.__doc__)

                    finc.write("\n")

                    try:
                        params = mprob.PROBLEM_PARAMS
                    except AttributeError:
                        params = {}

                    if params:
                        finc.write("parameters: \n\n")

                        finc.write("="*MAX_LEN + " " + "="*MAX_LEN + "\n")
                        finc.write(f"{'name':{MAX_LEN}} {'default':{MAX_LEN}}\n")
                        finc.write("="*MAX_LEN + " " + "="*MAX_LEN + "\n")

                        for k, v in params.items():
                            pname = "``" + k + "``"
                            finc.write(f"{pname:{MAX_LEN}} {v:{MAX_LEN}}\n")

                        finc.write("="*MAX_LEN + " " + "="*MAX_LEN + "\n")


                    finc.write("\n\n")

            else:
                finc.write(f"``{s}`` uses the problems defined by ``{parent_solver}``.")


if __name__ == "__main__":
    doit("..")
