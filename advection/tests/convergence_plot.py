import numpy
import pylab

def plot_convergence():

    data = numpy.loadtxt("smooth-error.out")

    nx = data[:,0]
    aerr = data[:,1]

    ax = pylab.subplot(111)
    ax.set_xscale('log')
    ax.set_yscale('log')

    pylab.scatter(nx, aerr, marker="x", color="r")
    pylab.plot(nx, aerr[0]*(nx[0]/nx)**2, "--", color="k")

    pylab.xlabel("number of zones")
    pylab.ylabel("L2 norm of abs error")

    pylab.title(r"convergence for smooth advection problem", fontsize=11)

    f = pylab.gcf()
    f.set_size_inches(5.0,5.0)

    pylab.xlim(8,256)

    pylab.savefig("smooth_converge.eps", bbox_inches="tight")

    

if __name__== "__main__":
    plot_convergence()

