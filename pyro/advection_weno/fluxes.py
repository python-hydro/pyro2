import numpy as np

import pyro.mesh.reconstruction as reconstruction


def fvs(q, order, u, alpha):
    """
    Perform Flux-Vector-Split (LF) finite differencing using WENO in 1d.

    Parameters
    ----------

    q : np array
        input data with at least order+1 ghost zones
    order : int
        WENO order (k)
    u : float
        Advection velocity in this direction
    alpha : float
        Maximum characteristic speed

    Returns
    -------

    f : np array
        flux
    """
    flux = u * q
    flux_p = (flux + alpha * q) / 2
    flux_m = (flux - alpha * q) / 2
    flux_p_r = np.zeros_like(flux_p)
    flux_m_l = np.zeros_like(flux_m)
    Npoints = len(q)
    for i in range(order, Npoints-order):
        flux_p_r[i] = reconstruction.weno_upwind(flux_p[i-order:i+order-1],
                                                 order)
        flux_m_l[i] = reconstruction.weno_upwind(flux_m[i+order-1:i-order:-1],
                                                 order)
    flux[1:-1] = flux_p_r[1:-1] + flux_m_l[1:-1]

    return flux


def fluxes(my_data, rp, dt):
    r"""
    Construct the fluxes through the interfaces for the linear advection
    equation

    .. math::

      a_t  + u a_x  + v a_y  = 0

    We use a high-order flux split WENO method to construct the interface
    fluxes. No Riemann problems are solved. The Lax-Friedrichs flux split
    will probably make it diffusive; the lack of a transverse solver also
    will not help.

    Parameters
    ----------
    my_data : CellCenterData2d object
        The data object containing the grid and advective scalar that
        we are advecting.
    rp : RuntimeParameters object
        The runtime parameters for the simulation
    dt : float
        The timestep we are advancing through.
    scalar_name : str
        The name of the variable contained in my_data that we are
        advecting

    Returns
    -------
    out : ndarray, ndarray
        The fluxes on the x- and y-interfaces

    """

    myg = my_data.grid

    a = my_data.get_var("density")

    # get the advection velocities
    u = rp.get_param("advection.u")
    v = rp.get_param("advection.v")

    qx = myg.qx
    qy = myg.qy

    # --------------------------------------------------------------------------
    # WENO fvs
    # --------------------------------------------------------------------------

    weno_order = rp.get_param("advection.weno_order")
    assert weno_order in (2, 3), "Currently only implemented weno_order=2, 3"
    assert myg.ng > weno_order, "Need more ghosts than the weno_order"

    q = a.v(buf=myg.ng)[:, :]
    F_x = myg.scratch_array()
    F_y = myg.scratch_array()

    alpha = np.sqrt(u**2 + v**2)

    # x-direction
    for j in range(qy):
        F_x.v(buf=myg.ng)[1:-1, j] = fvs(q[:, j], weno_order, u, alpha)[1:-1]
    # y-direction
    for i in range(qx):
        F_y.v(buf=myg.ng)[i, 1:-1] = fvs(q[i, :], weno_order, v, alpha)[1:-1]

    return F_x, F_y
