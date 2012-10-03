# test the boundary fill routine

import numpy
import mesh.patch
import pylab
import time

myg = mesh.patch.grid2d(4,4, ng = 2, xmax=1.0, ymax=1.0)

mydata = mesh.patch.ccData2d(myg, dtype=numpy.int)

bco = mesh.patch.bcObject(xlb="outflow", xrb="outflow",
                          ylb="outflow", yrb="outflow")
mydata.registerVar("outflow", bco)

bcp = mesh.patch.bcObject(xlb="periodic", xrb="periodic",
                          ylb="periodic", yrb="periodic")
mydata.registerVar("periodic", bcp)

bcre = mesh.patch.bcObject(xlb="reflect-even", xrb="reflect-even",
                           ylb="reflect-even", yrb="reflect-even")
mydata.registerVar("reflect-even", bcre)

bcro = mesh.patch.bcObject(xlb="reflect-odd", xrb="reflect-odd",
                           ylb="reflect-odd", yrb="reflect-odd")
mydata.registerVar("reflect-odd", bcro)

mydata.create()


a = mydata.getVarPtr("outflow")

i = myg.ilo
while (i <= myg.ihi):

    j = myg.jlo
    while (j <= myg.jhi):
        a[i,j] = (i-myg.ilo) + 10*(j-myg.jlo) + 1
        j += 1
    i += 1


b = mydata.getVarPtr("periodic")
c = mydata.getVarPtr("reflect-even")
d = mydata.getVarPtr("reflect-odd")

b[:,:] = a[:,:]
c[:,:] = a[:,:]
d[:,:] = a[:,:]

mydata.fillBC("outflow")
mydata.fillBC("periodic")
mydata.fillBC("reflect-even")
mydata.fillBC("reflect-odd")



print "outflow"
mydata.prettyPrint("outflow")

print " "
print "periodic"
mydata.prettyPrint("periodic")

print " "
print "reflect-even"
mydata.prettyPrint("reflect-even")

print " "
print "reflect-odd"
mydata.prettyPrint("reflect-odd")

