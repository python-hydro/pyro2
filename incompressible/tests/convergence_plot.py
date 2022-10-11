import matplotlib.pyplot as plt
import numpy as np


def plot_convergence():

    data = np.loadtxt("incomp_converge.txt")

    nx = data[:, 0]
    errlim = data[:, 1]

    ax = plt.subplot(111)
    ax.set_xscale('log')
    ax.set_yscale('log')

    plt.scatter(nx, errlim, marker="x", color="r", label="limiting")
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
