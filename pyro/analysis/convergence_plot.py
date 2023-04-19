#!/usr/bin/env python3

import argparse
import sys

import convergence
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable

import pyro.util.io_pyro as io

description = "This file does convergence plotting for the solvers"
input_help = '''The location of the input files: enter from finest to coarset files,
at least 3 files are required'''
output_help = "File name for the convergence plot"
order_help = "The theoretical order of convergence for the solver"
resolution_help = "Multiplicative Resolution between input files, default is 2"

parser = argparse.ArgumentParser(description=description)
parser.add_argument("input_file", nargs='+', help=input_help)
parser.add_argument("-o", "--out", default="convergence_plot.pdf", help=output_help)
parser.add_argument("-n", "--order", default=2, type=int, help=order_help)
parser.add_argument("-r", "--resolution", default=2, type=int, help=resolution_help)

args = parser.parse_args(sys.argv[1:])


def convergence_plot(dataset, fname=None, order=2):
    # Plot the error and its theoretical convergence line.

    figure = plt.figure(figsize=(12, 7))

    tot = len(dataset)
    cols = 2
    rows = tot // 2

    if tot % cols != 0:
        rows += 1

    for n, data in enumerate(dataset):

        ax = figure.add_subplot(rows, cols, n+1)
        ax.set(
            title=f"{data[0]}",
            xlabel="$N$",
            ylabel="L2 Norm of Error",
            yscale="log",
            xscale="log"
        )

        err = np.array(data[1])
        N = np.array(data[2])

        ax.scatter(N, err, marker='x', color='k', label="Solutions")
        ax.plot(N, err[0]*(N[0]/N)**2, linestyle="--", label=r"$\mathcal{O}$" + fr"$(\Delta x^{order})$")

        ax.legend()

    plt.tight_layout()
    plt.show()

    if fname is not None:
        plt.savefig(fname, format="pdf", bbox_inches="tight")


if __name__ == "__main__":

    if len(args.input_file) < 3:
        raise ValueError('''Must use at least 3 plotfiles with 3 different
        resolutions that differ by the same multiplicative factor''')

    fine = io.read(args.input_file[0])
    med = io.read(args.input_file[1])
    coarse = io.read(args.input_file[2])

    field_names = ["Variable Name", "L2 Norm (Finest)"]

    for i in range(len(args.input_file[2:])):
        field_names.append("Order of Conv")
        field_names.append("L2 Norm")

    field_names[-1] += " (Coarsest)"

    table = PrettyTable()
    table._validate_field_names = lambda *a, **k: None
    table.field_names = field_names

    dataset = []
    l2norms = []
    ns = []

    for variable in fine.cc_data.names:

        l2norms = []
        ns = []
        row = [variable]

        fdata = io.read(args.input_file[0])
        cdata = io.read(args.input_file[1])
        _, l2norm = convergence.compare(fdata.cc_data, cdata.cc_data, variable, args.resolution)

        l2norms.append(l2norm)
        ns.append(cdata.cc_data.grid.nx)
        row.append(l2norm)

        for n in range(1, len(args.input_file[:-1])):
            fdata = io.read(args.input_file[n])
            cdata = io.read(args.input_file[n+1])

            _, l2norm = convergence.compare(fdata.cc_data, cdata.cc_data, variable, args.resolution)

            order = np.sqrt(l2norm/l2norms[-1])

            l2norms.append(l2norm)
            ns.append(cdata.cc_data.grid.nx)
            row.extend([order, l2norm])

        dataset.append([variable, l2norms, ns])
        table.add_row(row)

    print(table)
    convergence_plot(dataset, fname=args.out, order=args.order)
