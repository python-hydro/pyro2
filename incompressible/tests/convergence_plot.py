import math
import numpy
import pylab

def plot_convergence():

    data = numpy.loadtxt("incomp_converge.txt")

    nx = data[:,0]
    errlim = data[:,1]
    errnolim = data[:,3]

    ax = pylab.subplot(111)
    ax.set_xscale('log')
    ax.set_yscale('log')

    pylab.scatter(nx, errlim, marker="x", color="r", label="limiting")
    pylab.plot(nx, errlim[0]*(nx[0]/nx)**2, "--", color="k")

    pylab.scatter(nx, errnolim, marker="x", color="b", label="no limiting")
    pylab.plot(nx, errnolim[0]*(nx[0]/nx)**2, "--", color="k")

    pylab.xlabel("number of zones")
    pylab.ylabel("L2 norm of abs error")

    pylab.title(r"convergence for multigrid solver", fontsize=11)

    f = pylab.gcf()
    f.set_size_inches(5.0,5.0)

    pylab.xlim(16,256)
    pylab.ylim(2.e-4,5.e-2)

    leg = pylab.legend()
    leg.draw_frame(0)
    ltext = leg.get_texts()
    pylab.setp(ltext, fontsize='small')

    pylab.savefig("incomp_converge.png", bbox_inches="tight")

    

if __name__== "__main__":
    plot_convergence()

