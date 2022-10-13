#!/usr/bin/env python3

"""test of a cell-centered, centered-difference approximate projection.

initialize the velocity field to be divergence free and then add to it
the gradient of a scalar (whose normal component vanishes on the
boundaries).  The projection should recover the original divergence-
free velocity field.

The test velocity field comes from Almgen, Bell, and Szymczak 1996.

This makes use of the multigrid solver with periodic boundary conditions.

One of the things that this test demonstrates is that the initial
projection may not be able to completely remove the divergence free
part, so subsequent projections may be necessary.  In this example, we
add a very strong gradient component.

The total number of projections to perform is given by nproj.  Each
projection uses the divergence of the velocity field from the previous
iteration as its source term.

Note: the output file created stores the original field, the poluted
field, and the recovered field.
"""

import numpy as np

import pyro.mesh.boundary as bnd
import pyro.mesh.patch as patch
import pyro.multigrid.MG as MG


def doit(nx, ny):
    """manage the entire projection"""

    nproj = 2

    # create a mesh containing the x- and y-velocities, and periodic boundary
    # conditions
    myg = patch.Grid2d(nx, ny, ng=1)

    bc = bnd.BC(xlb="periodic", xrb="periodic",
                ylb="periodic", yrb="periodic")

    U = patch.CellCenterData2d(myg)

    U.register_var('u-old', bc)
    U.register_var('v-old', bc)
    U.register_var('u+gphi', bc)
    U.register_var('v+gphi', bc)
    U.register_var('u', bc)
    U.register_var('v', bc)

    U.register_var('divU', bc)

    U.register_var('phi-old', bc)
    U.register_var('phi', bc)
    U.register_var('dphi', bc)

    U.register_var('gradphi_x-old', bc)
    U.register_var('gradphi_y-old', bc)
    U.register_var('gradphi_x', bc)
    U.register_var('gradphi_y', bc)

    U.create()

    # initialize a divergence free velocity field,
    # u = -sin^2(pi x) sin(2 pi y), v = sin^2(pi y) sin(2 pi x)
    u = U.get_var('u')
    v = U.get_var('v')

    u[:, :] = -(np.sin(np.pi*myg.x2d)**2)*np.sin(2.0*np.pi*myg.y2d)
    v[:, :] = (np.sin(np.pi*myg.y2d)**2)*np.sin(2.0*np.pi*myg.x2d)

    # store the original, divergence free velocity field for comparison later
    uold = U.get_var('u-old')
    vold = U.get_var('v-old')

    uold[:, :] = u.copy()
    vold[:, :] = v.copy()

    # the projection routine should decompose U into a divergence free
    # part, U_d, plus the gradient of a scalar.  Add on the gradient
    # of a scalar that satisfies gradphi.n = 0.  After the projection,
    # we should recover the divergence free field above.  Take phi to
    # be a gaussian, exp(-((x-x0)^2 + (y-y0)^2)/R)
    R = 0.1
    x0 = 0.5
    y0 = 0.5

    phi = U.get_var('phi-old')
    gradphi_x = U.get_var('gradphi_x-old')
    gradphi_y = U.get_var('gradphi_y-old')

    phi[:, :] = np.exp(-((myg.x2d-x0)**2 + (myg.y2d-y0)**2)/R**2)

    gradphi_x[:, :] = phi*(-2.0*(myg.x2d-x0)/R**2)
    gradphi_y[:, :] = phi*(-2.0*(myg.y2d-y0)/R**2)

    u += gradphi_x
    v += gradphi_y

    u_plus_gradphi = U.get_var('u+gphi')
    v_plus_gradphi = U.get_var('v+gphi')

    u_plus_gradphi[:, :] = u[:, :]
    v_plus_gradphi[:, :] = v[:, :]

    # use the mesh class to enforce the periodic BCs on the velocity field
    U.fill_BC_all()

    # now compute the cell-centered, centered-difference divergence
    divU = U.get_var('divU')

    divU[myg.ilo:myg.ihi+1, myg.jlo:myg.jhi+1] = \
          0.5*(u[myg.ilo+1:myg.ihi+2, myg.jlo:myg.jhi+1] -
               u[myg.ilo-1:myg.ihi, myg.jlo:myg.jhi+1])/myg.dx + \
          0.5*(v[myg.ilo:myg.ihi+1, myg.jlo+1:myg.jhi+2] -
               v[myg.ilo:myg.ihi+1, myg.jlo-1:myg.jhi])/myg.dy

    # create the multigrid object with Neumann BCs
    a = MG.CellCenterMG2d(nx, ny,
                          xl_BC_type="periodic", xr_BC_type="periodic",
                          yl_BC_type="periodic", yr_BC_type="periodic",
                          verbose=1)

    # --------------------------------------------------------------------------
    # projections
    # --------------------------------------------------------------------------
    for iproj in range(nproj):

        a.init_zeros()
        a.init_RHS(divU)
        a.solve(rtol=1.e-12)

        phi = U.get_var('phi')
        solution = a.get_solution()

        phi[myg.ilo-1:myg.ihi+2, myg.jlo-1:myg.jhi+2] = \
            solution[a.ilo-1:a.ihi+2, a.jlo-1:a.jhi+2]

        dphi = U.get_var('dphi')
        dphi[:, :] = phi - U.get_var('phi-old')

        # compute the gradient of phi using centered differences
        gradphi_x = U.get_var('gradphi_x')
        gradphi_y = U.get_var('gradphi_y')

        gradphi_x[myg.ilo:myg.ihi+1, myg.jlo:myg.jhi+1] = \
            0.5*(phi[myg.ilo+1:myg.ihi+2, myg.jlo:myg.jhi+1] -
                 phi[myg.ilo-1:myg.ihi, myg.jlo:myg.jhi+1])/myg.dx

        gradphi_y[myg.ilo:myg.ihi+1, myg.jlo:myg.jhi+1] = \
            0.5*(phi[myg.ilo:myg.ihi+1, myg.jlo+1:myg.jhi+2] -
                 phi[myg.ilo:myg.ihi+1, myg.jlo-1:myg.jhi])/myg.dy

        # update the velocity field
        u -= gradphi_x
        v -= gradphi_y

        U.fill_BC_all()

        # recompute the divergence diagnostic
        divU[myg.ilo:myg.ihi+1, myg.jlo:myg.jhi+1] = \
             0.5*(u[myg.ilo+1:myg.ihi+2, myg.jlo:myg.jhi+1] -
                  u[myg.ilo-1:myg.ihi, myg.jlo:myg.jhi+1])/myg.dx + \
             0.5*(v[myg.ilo:myg.ihi+1, myg.jlo+1:myg.jhi+2] -
                  v[myg.ilo:myg.ihi+1, myg.jlo-1:myg.jhi])/myg.dy

        U.write("proj-periodic.after"+("%d" % iproj))


if __name__ == "__main__":
    doit(128, 128)
