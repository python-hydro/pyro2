import patch

import numpy as np
import time

myg = patch.Grid2d(512,1024, xmax=1.0, ymax=2.0)

myd = patch.CellCenterData2d(myg)

bc = patch.BCObject()
myd.register_var("a", bc)
myd.create()

a = myd.get_var("a")
a[:,:] = np.random.rand(myg.qx, myg.qy)


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
