# test the boundary fill routine

import numpy as np
import mesh.patch

myg = mesh.patch.Grid2d(4,4, ng = 2, xmax=1.0, ymax=1.0)

mydata = mesh.patch.CellCenterData2d(myg, dtype=np.int)

bco = mesh.patch.BCObject(xlb="outflow", xrb="outflow",
                          ylb="outflow", yrb="outflow")
mydata.register_var("outflow", bco)

bcp = mesh.patch.BCObject(xlb="periodic", xrb="periodic",
                          ylb="periodic", yrb="periodic")
mydata.register_var("periodic", bcp)

bcre = mesh.patch.BCObject(xlb="reflect-even", xrb="reflect-even",
                           ylb="reflect-even", yrb="reflect-even")
mydata.register_var("reflect-even", bcre)

bcro = mesh.patch.BCObject(xlb="reflect-odd", xrb="reflect-odd",
                           ylb="reflect-odd", yrb="reflect-odd")
mydata.register_var("reflect-odd", bcro)

mydata.create()


a = mydata.get_var("outflow")

for i in range(myg.ilo, myg.ihi+1):
    for j in range(myg.jlo, myg.jhi+1):
        a.d[i,j] = (i-myg.ilo) + 10*(j-myg.jlo) + 1


b = mydata.get_var("periodic")
c = mydata.get_var("reflect-even")
d = mydata.get_var("reflect-odd")

b.d[:,:] = a.d[:,:]
c.d[:,:] = a.d[:,:]
d.d[:,:] = a.d[:,:]

mydata.fill_BC("outflow")
mydata.fill_BC("periodic")
mydata.fill_BC("reflect-even")
mydata.fill_BC("reflect-odd")



print "outflow"
mydata.pretty_print("outflow")

print " "
print "periodic"
mydata.pretty_print("periodic")

print " "
print "reflect-even"
mydata.pretty_print("reflect-even")

print " "
print "reflect-odd"
mydata.pretty_print("reflect-odd")

