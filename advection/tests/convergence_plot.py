import math
import numpy
import pylab

def plot_convergence():

    data = numpy.loadtxt("advection_convergence.txt")

    nx = data[:,0]
    aerr_lim1 = data[:,1]
    aerr_lim2 = data[:,3]

    ax = pylab.subplot(111)
    ax.set_xscale('log')
    ax.set_yscale('log')

    pylab.scatter(nx, aerr_lim2, marker="x", color="r")
    pylab.plot(nx, aerr_lim2[0]*(nx[0]/nx)**2, "--", color="k")

    pylab.xlabel("number of zones")
    pylab.ylabel("L2 norm of abs error")

    pylab.title(r"convergence for smooth advection problem", fontsize=11)

    f = pylab.gcf()
    f.set_size_inches(5.0,5.0)

    pylab.xlim(8,256)

    pylab.savefig("smooth_converge.png", bbox_inches="tight")

    

if __name__== "__main__":
    plot_convergence()

