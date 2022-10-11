import matplotlib.pyplot as plt
import numpy as np


def plot_convergence():

    data4 = np.loadtxt("advection_fv4_convergence.txt")

    nx = data4[:, 0]
    aerr = data4[:, 1]

    ax = plt.subplot(111)
    ax.set_xscale('log')
    ax.set_yscale('log')

    plt.scatter(nx, aerr, marker="x", color="C1", label="4th-order advection solver")
    plt.plot(nx, aerr[0]*(nx[0]/nx)**4, "--", color="C0", label=r"$\mathcal{O}(\Delta x^4)$")

    data2 = np.loadtxt("../../advection/tests/advection_convergence.txt")

    nx = data2[:, 0]
    aerr = data2[:, 1]

    ax = plt.subplot(111)
    ax.set_xscale('log')
    ax.set_yscale('log')

    plt.scatter(nx, aerr, marker="o", color="C1", label="2nd-order advection solver")
    plt.plot(nx, aerr[0]*(nx[0]/nx)**2, ":", color="C0", label=r"$\mathcal{O}(\Delta x^2)$")

    plt.xlabel("number of zones")
    plt.ylabel("L2 norm of abs error")

    plt.legend(frameon=False, fontsize="small")

    plt.title(r"convergence for smooth advection problem", fontsize=11)

    fig = plt.gcf()
    fig.set_size_inches(6.0, 5.0)

    plt.xlim(8, 256)

    plt.savefig("smooth_converge.png", bbox_inches="tight", dpi=150)


if __name__ == "__main__":
    plot_convergence()
