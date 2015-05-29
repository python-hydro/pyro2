import patch

import numpy as np
import time

n = 1024

myg = patch.Grid2d(n,2*n, xmax=1.0, ymax=2.0)

myd = patch.CellCenterData2d(myg)

bc = patch.BCObject()
myd.register_var("a", bc)
myd.create()

a = myd.get_var("a")
a[:,:] = np.random.rand(myg.qx, myg.qy)
#a[:,:] = np.arange(myg.qx*myg.qy).reshape(myg.qx, myg.qy)

# slicing method
start = time.time()

da = myg.scratch_array()
da[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] = \
    a[myg.ilo+1:myg.ihi+2,myg.jlo:myg.jhi+1] - \
    a[myg.ilo-1:myg.ihi  ,myg.jlo:myg.jhi+1]

print "slice method: ", time.time() - start

                            
# mask method
m = patch.Mask(myg.nx, myg.ny, myg.ng)

start = time.time()
da2 = myg.scratch_array()
da2[m.valid] = a[m.ip1] - a[m.im1]

print "mask method: ", time.time() - start

print np.max(np.abs(da2 - da))

# roll -- note, we roll in the opposite direction of the shift
start = time.time()
da3 = myg.scratch_array()
da3[:] = np.roll(a, -1, axis=0) - np.roll(a, 1, axis=0)

print "roll method: ", time.time() - start

print np.max(np.abs(da3[m.valid] - da[m.valid]))


# ArrayIndex
start = time.time()
ai = patch.ArrayIndex(d=a, grid=myg)
da4 = myg.scratch_array()
da4i = patch.ArrayIndex(d=da4, grid=myg)

da4i.v()[:,:] = ai.ip(1) - ai.ip(-1)

print "ArrayIndex method: ", time.time() - start

print np.max(np.abs(da4[m.valid] - da[m.valid]))
