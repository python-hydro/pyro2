import os

import matplotlib.pyplot as plt
import numpy as np


def plot_convergence():

    ax = plt.subplot(111)
    ax.set_xscale('log')
    ax.set_yscale('log')

    files = ["incompressible/tests/convergence_errors.txt",
             "incompressible/tests/convergence_errors_no_limiter.txt"]

    if not os.path.exists(files[0]):
        print("Could not find convergence_errors.txt")
    else:
        data = np.loadtxt(files[0])
        nx = data[:, 0]
        errlim = data[:, 1]

        plt.scatter(nx, errlim, marker="x", color="r", label="limiting")
        plt.plot(nx, errlim[0]*(nx[0]/nx)**2, "--", color="k")

    if not os.path.exists(files[1]):
        print("Could not find convergence_errors_no_limiter.txt")
    else:
        data = np.loadtxt(files[1])
        nx = data[:, 0]
        errlim = data[:, 1]

        plt.scatter(nx, errlim, marker="x", color="b", label="limiting")
        plt.plot(nx, errlim[0]*(nx[0]/nx)**2, "--", color="k")

    plt.xlabel("number of zones")
    plt.ylabel("L2 norm of abs error")

    plt.title(r"convergence for incompressible solver", fontsize=11)

    f = plt.gcf()
    f.set_size_inches(5.0, 5.0)

    plt.xlim(16, 256)
    plt.ylim(2.e-4, 5.e-2)

    plt.savefig("incomp_converge.png", bbox_inches="tight")
    plt.savefig("incomp_converge.eps", bbox_inches="tight")


if __name__ == "__main__":
    plot_convergence()
