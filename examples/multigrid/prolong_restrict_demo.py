# test the prolongation and restriction operations from the patch stuff


import numpy as np

import pyro.mesh.boundary as bnd
import pyro.mesh.patch as patch


def doit():

    # create our base grid and initialize it with sequential data
    myg = patch.Grid2d(4, 8, ng=1)
    myd = patch.CellCenterData2d(myg)
    bc = bnd.BC()
    myd.register_var("a", bc)
    myd.create()

    a = myd.get_var("a")

    a[myg.ilo:myg.ihi+1, myg.jlo:myg.jhi+1].flat = np.arange(myg.nx*myg.ny)

    print("restriction test")
    print("original (fine) array")
    myd.pretty_print("a")

    # create a coarse grid and fill the variable in it with restricted data
    print(" ")
    print("restricted array")

    cg = patch.Grid2d(2, 4, ng=1)
    cd = patch.CellCenterData2d(cg)
    cd.register_var("a", bc)
    cd.create()

    a_coarse = cd.get_var("a")
    a_coarse[:, :] = myd.restrict("a")[:, :]

    cd.pretty_print("a")

    print(" ")
    print("prolongation test")
    print("original (coarse) array w/ ghost cells")
    a_coarse[:, :].flat = np.arange(cg.qx*cg.qy)
    cd.pretty_print("a")

    # create a new fine (base) grid and fill the variable in it prolonged data
    # from the coarsened grid
    print(" ")
    print("prolonged array")

    fg = patch.Grid2d(4, 8, ng=1)
    fd = patch.CellCenterData2d(fg)
    fd.register_var("a", bc)
    fd.create()

    a_fine = fd.get_var("a")
    a_fine[:, :] = cd.prolong("a")[:, :]

    fd.pretty_print("a")


if __name__ == "__main__":
    doit()
