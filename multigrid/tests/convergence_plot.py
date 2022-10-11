import matplotlib.pyplot as plt
import numpy as np


def plot_convergence():

    data = np.loadtxt("mg_convergence.txt")

    nx = data[:, 0]
    err = data[:, 1]

    ax = plt.subplot(111)
    ax.set_xscale('log')
    ax.set_yscale('log')

    plt.scatter(nx, err, marker="x", color="r")
    plt.plot(nx, err[0]*(nx[0]/nx)**2, "--", color="k")

    plt.xlabel("number of zones")
    plt.ylabel("L2 norm of abs error")

    plt.title(r"convergence for multigrid solver", fontsize=11)

    f = plt.gcf()
    f.set_size_inches(5.0, 5.0)

    plt.xlim(8, 512)

    plt.tight_layout()

    plt.savefig("mg_converge.png", bbox_inches="tight")
    plt.savefig("mg_converge.eps", bbox_inches="tight")


if __name__ == "__main__":
    plot_convergence()
