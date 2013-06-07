import math
import numpy
import pylab

def plot_convergence():

    data = numpy.loadtxt("mg_convergence.txt")

    nx = data[:,0]
    err = data[:,1]

    ax = pylab.subplot(111)
    ax.set_xscale('log')
    ax.set_yscale('log')

    pylab.scatter(nx, err, marker="x", color="r")
    pylab.plot(nx, err[0]*(nx[0]/nx)**2, "--", color="k")

    pylab.xlabel("number of zones")
    pylab.ylabel("L2 norm of abs error")

    pylab.title(r"convergence for multigrid solver", fontsize=11)

    f = pylab.gcf()
    f.set_size_inches(5.0,5.0)

    pylab.xlim(8,512)

    pylab.tight_layout()

    pylab.savefig("mg_converge.png", bbox_inches="tight")
    pylab.savefig("mg_converge.eps", bbox_inches="tight")

    

if __name__== "__main__":
    plot_convergence()

