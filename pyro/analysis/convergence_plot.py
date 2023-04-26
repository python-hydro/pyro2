#!/usr/bin/env python3

'''
This file prints summary of the convergence given at least 3 plot files
that are differ by a constant multiplicative resolution factor.
It prints out a table as well as a plot.
'''

import argparse
import sys

import convergence
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable

import pyro.util.io_pyro as io

DESCRIPTION = "This file does convergence plotting for the solvers"
INPUT_HELP = '''The location of the input files: enter from finest to coarset files,
at least 3 files are required'''
OUTPUT_HELP = "File name for the convergence plot"
ORDER_HELP = "The theoretical order of convergence for the solver"
RESOLUTION_HELP = "Multiplicative Resolution between input files, default is 2"

parser = argparse.ArgumentParser(description=DESCRIPTION)
parser.add_argument("input_file", nargs='+', help=INPUT_HELP)
parser.add_argument("-o", "--out", default="convergence_plot.pdf", help=OUTPUT_HELP)
parser.add_argument("-n", "--order", default=2, type=int, help=ORDER_HELP)
parser.add_argument("-r", "--resolution", default=2, type=int, help=RESOLUTION_HELP)

args = parser.parse_args(sys.argv[1:])


def convergence_plot(dataset, fname=None, order=2):
    ''' Plot the error and its theoretical convergence line. '''

    figure = plt.figure(figsize=(12, 7))

    tot = len(dataset)
    cols = 2
    rows = tot // 2

    if tot % cols != 0:
        rows += 1

    for k, data in enumerate(dataset):

        ax = figure.add_subplot(rows, cols, k+1)
        ax.set(
            title=f"{data[0]}",
            xlabel="$N$",
            ylabel="L2 Norm of Error",
            yscale="log",
            xscale="log"
        )

        err = np.array(data[1])
        nx = np.array(data[2])

        ax.scatter(nx, err, marker='x', color='k', label="Solutions")
        ax.plot(nx, err[-1]*(nx[-1]/nx)**2, linestyle="--",
                label=r"$\mathcal{O}$" + fr"$(\Delta x^{order})$")

        ax.legend()

    plt.tight_layout()

    if fname is not None:
        plt.savefig(fname, format="pdf", bbox_inches="tight")

    plt.show()


def main():
    ''' main driver '''

    if len(args.input_file) < 3:
        raise ValueError('''Must use at least 3 plotfiles with 3 different
        resolutions that differ by the same multiplicative factor''')

    field_names = ["Variable Name", "L2 Norm(1) (Finest)"]

    for i in range(len(args.input_file[2:])):
        field_names.append(f"Order of Conv({i+1}:{i+2})")
        field_names.append(f"L2 Norm({i+2})")

    field_names[-1] += " (Coarsest)"

    table = PrettyTable(field_names)

    data_file = []

    temp_file = io.read(args.input_file[0])
    for variable in temp_file.cc_data.names:

        l2norms = []
        nx_list = []
        row = [variable]

        fdata = io.read(args.input_file[0])
        cdata = io.read(args.input_file[1])
        _, l2norm = convergence.compare(fdata.cc_data, cdata.cc_data, variable, args.resolution)

        l2norms.append(l2norm)
        nx_list.append(cdata.cc_data.grid.nx)
        row.append(l2norm)

        for n in range(1, len(args.input_file[:-1])):
            fdata = io.read(args.input_file[n])
            cdata = io.read(args.input_file[n+1])

            _, l2norm = convergence.compare(fdata.cc_data, cdata.cc_data, variable, args.resolution)

            order_conv = np.sqrt(l2norm/l2norms[-1])

            l2norms.append(l2norm)
            nx_list.append(cdata.cc_data.grid.nx)
            row.extend([order_conv, l2norm])

        data_file.append([variable, l2norms, nx_list])
        table.add_row(row)

    print(table)
    convergence_plot(data_file, fname=args.out, order=args.order)


if __name__ == "__main__":
    main()
