# illustrate basic mesh operations

# note for interactive plots, we need to have the matplotlib backend
# set to TkAgg.  In ~/.matplotlib/matplotlibrc, set:
#
# backend: TkAgg


import numpy
import mesh.patch
import pylab
import time

myg = mesh.patch.grid2d(16,32, xmax=1.0, ymax=2.0)

mydata = mesh.patch.ccData2d(myg)

bc = mesh.patch.bcObject()

mydata.registerVar("a", bc)
mydata.create()

print mydata.__class__.__name__

a = mydata.getVarPtr("a")

print type(a)

a[:,:] = numpy.exp(-(myg.x2d - 0.5)**2 - (myg.y2d - 1.0)**2)

pylab.ion()

pylab.clf()

# make a plot -- note that in python, the rightmost index is 
# contiguous in memory, and the plotting routines interpret this
# as the 'x' axis, so we transpose here.
pylab.imshow(numpy.transpose(a), interpolation="nearest", origin="lower", 
             extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax])

pylab.draw()

pylab.show()


n = 0
while (n < myg.nx):
    print n
    a[:,:] = numpy.exp(-(myg.x2d - 0.5 + n*myg.dx)**2 - (myg.y2d - 1.0)**2)

    pylab.clf()
    pylab.imshow(numpy.transpose(a), interpolation="nearest", origin="lower",
             extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax])

    pylab.draw()
    
    time.sleep(5)

    n += 1


print "here"

