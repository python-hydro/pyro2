import matplotlib.pyplot as plt
import numpy as np


def plot_convergence():

    data = np.loadtxt("smooth-error.out")

    nx = data[:, 0]
    aerr = data[:, 1]

    ax = plt.subplot(111)
    ax.set_xscale('log')
    ax.set_yscale('log')

    plt.scatter(nx, aerr, marker="x", color="r")
    plt.plot(nx, aerr[0]*(nx[0]/nx)**2, "--", color="k")

    plt.xlabel("number of zones")
    plt.ylabel("L2 norm of abs error")

    plt.title(r"convergence for smooth advection problem", fontsize=11)

    f = plt.gcf()
    f.set_size_inches(5.0, 5.0)

    plt.xlim(8, 256)

    plt.savefig("smooth_converge.eps", bbox_inches="tight")


if __name__ == "__main__":
    plot_convergence()
