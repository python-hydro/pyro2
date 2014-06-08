# test the prolongation and restriction operations from the patch stuff

import mesh.patch as patch
import numpy

# create our base grid and initialize it with sequential data
myg = patch.Grid2d(4,8,ng=1)
myd = patch.CellCenterData2d(myg)
bc = patch.BCObject()
myd.registerVar("a", bc)
myd.create()

a = myd.getVarPtr("a")

a[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1].flat = numpy.arange(myg.nx*myg.ny)

print "restriction test"
print "original (fine) array"
myd.prettyPrint("a")


# create a coarse grid and fill the variable in it with restricted data
print " "
print "restricted array"

cg = patch.Grid2d(2,4,ng=1)
cd = patch.CellCenterData2d(cg)
cd.registerVar("a", bc)
cd.create()

a_coarse = cd.getVarPtr("a")
a_coarse[:,:] = myd.restrict("a")

cd.prettyPrint("a")


print " "
print "prolongation test"
print "original (coarse) array w/ ghost cells"
a_coarse[:,:].flat = numpy.arange(cg.qx*cg.qy)
cd.prettyPrint("a")


# create a new fine (base) grid and fill the variable in it prolonged data
# from the coarsened grid
print " "
print "prolonged array"

fg = patch.Grid2d(4,8,ng=1)
fd = patch.CellCenterData2d(fg)
fd.registerVar("a", bc)
fd.create()

a_fine = fd.getVarPtr("a")
a_fine[:,:] = cd.prolong("a")

fd.prettyPrint("a")





