#!/usr/bin/env python3

import time

import numpy as np

import mesh.boundary as bnd
from mesh import patch


class Mask(object):

    def __init__(self, nx, ny, ng):

        self.nx = nx
        self.ny = ny
        self.ng = ng

        ilo = ng
        ihi = ng+nx-1

        jlo = ng
        jhi = ng+ny-1

        # just the interior cells
        self.valid = self._mask_array(nx, ny, ng)

        # shifts in x
        self.ip1 = self._mask_array(nx, ny, ng)
        self.im1 = self._mask_array(nx, ny, ng)
        self.ip2 = self._mask_array(nx, ny, ng)
        self.im2 = self._mask_array(nx, ny, ng)

        arrays = [self.valid, self.ip1, self.im1, self.ip2, self.im2]
        shifts = [0, 1, -1, 2, -2]

        for a, s in zip(arrays, shifts):
            a[ilo+s:ihi+1+s, jlo:jhi+1] = True

        # shifts in y
        self.jp1 = self._mask_array(nx, ny, ng)
        self.jm1 = self._mask_array(nx, ny, ng)
        self.jp2 = self._mask_array(nx, ny, ng)
        self.jm2 = self._mask_array(nx, ny, ng)

        arrays = [self.jp1, self.jm1, self.jp2, self.jm2]
        shifts = [1, -1, 2, -2]

        for a, s in zip(arrays, shifts):
            a[ilo:ihi+1, jlo+s:jhi+1+s] = True

    def _mask_array(self, nx, ny, ng):
        return np.zeros((nx+2*ng, ny+2*ng), dtype=bool)


n = 1024

myg = patch.Grid2d(n, 2*n, xmax=1.0, ymax=2.0)

myd = patch.CellCenterData2d(myg)

bc = bnd.BC()
myd.register_var("a", bc)
myd.create()

a = myd.get_var("a")
a[:, :] = np.random.rand(myg.qx, myg.qy)

# slicing method
start = time.time()

da = myg.scratch_array()
da[myg.ilo:myg.ihi+1, myg.jlo:myg.jhi+1] = \
    a[myg.ilo+1:myg.ihi+2, myg.jlo:myg.jhi+1] - \
    a[myg.ilo-1:myg.ihi, myg.jlo:myg.jhi+1]

print("slice method: ", time.time() - start)


# mask method
m = Mask(myg.nx, myg.ny, myg.ng)

start = time.time()
da2 = myg.scratch_array()
da2[m.valid] = a[m.ip1] - a[m.im1]

print("mask method: ", time.time() - start)

print(np.max(np.abs(da2 - da)))

# roll -- note, we roll in the opposite direction of the shift
start = time.time()
da3 = myg.scratch_array()
da3[:] = np.roll(a, -1, axis=0) - np.roll(a, 1, axis=0)

print("roll method: ", time.time() - start)

print(np.max(np.abs(da3[m.valid] - da[m.valid])))


# ArrayIndex
start = time.time()
da4 = myg.scratch_array()

da4.v()[:, :] = a.ip(1) - a.ip(-1)

print("ArrayIndex method: ", time.time() - start)

print(np.max(np.abs(da4[m.valid] - da[m.valid])))
