import numpy as np

import pyro.mesh.reconstruction as reconstruction


def riemann(my_data, ul, ur):

    myg = my_data.grid

    S = myg.scratch_array()
    S.v(buf=1)[:, :] = 0.5*(ul.v(buf=1)+ur.v(buf=1))

    # shock when ul > ur
    shock = myg.scratch_array()
    shock.v(buf=1)[:, :] = np.where(S.v(buf=1) > 0.0, ul.v(buf=1), shock.v(buf=1))
    shock.v(buf=1)[:, :] = np.where(S.v(buf=1) < 0.0, ur.v(buf=1), shock.v(buf=1))

    # rarefaction otherwise
    rarefac = myg.scratch_array()
    rarefac.v(buf=1)[:, :] = np.where(ul.v(buf=1) > 0.0, ul.v(buf=1), rarefac.v(buf=1))
    rarefac.v(buf=1)[:, :] = np.where(ur.v(buf=1) < 0.0, ur.v(buf=1), rarefac.v(buf=1))

    state = myg.scratch_array()

    # shock (compression) if the left interface state is faster than the right interface state
    state.v(buf=1)[:, :] = np.where(ul.v(buf=1) > ur.v(buf=1), shock.v(buf=1), rarefac.v(buf=1))

    return state


def unsplit_fluxes(my_data, rp, dt, scalar_name):
    r"""
    Construct the fluxes through the interfaces for the burgers equation:

    .. math::

       u_t  + u u_x  + v u_y  = 0
       v_t  + u v_x  + v v_y  = 0

    We use a second-order (piecewise linear) unsplit Godunov method
    (following Colella 1990).

    Our convection is that the fluxes are going to be defined on the
    left edge of the computational zones::

        |             |             |             |
        |             |             |             |
       -+------+------+------+------+------+------+--
        |     i-1     |      i      |     i+1     |

                 a_l,i  a_r,i   a_l,i+1


    a_r,i and a_l,i+1 are computed using the information in
    zone i,j.

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

    a = my_data.get_var(scalar_name)

    u = my_data.get_var("x-velocity")
    v = my_data.get_var("y-velocity")

    cx = myg.scratch_array()
    cy = myg.scratch_array()

    cx.v(buf=1)[:, :] = u.v(buf=1)*dt/myg.dx
    cy.v(buf=1)[:, :] = v.v(buf=1)*dt/myg.dy

    # --------------------------------------------------------------------------
    # monotonized central differences
    # --------------------------------------------------------------------------

    limiter = rp.get_param("advection.limiter")

    # Give da/dx and da/dy using input: (state, grid, direction, limiter)

    ldelta_ax = reconstruction.limit(a, myg, 1, limiter)
    ldelta_ay = reconstruction.limit(a, myg, 2, limiter)

    ul_x = myg.scratch_array()
    ur_x = myg.scratch_array()

    ul_y = myg.scratch_array()
    ur_y = myg.scratch_array()

    # Determine left and right interface states in x and y.

    # First compute the predictor terms

    ul_x.v(buf=1)[:, :] = a.ip(-1, buf=1) + 0.5*(1.0 - cx.ip(-1, buf=1))*ldelta_ax.ip(-1, buf=1)
    ul_y.v(buf=1)[:, :] = a.jp(-1, buf=1) + 0.5*(1.0 - cy.jp(-1, buf=1))*ldelta_ay.jp(-1, buf=1)
    ur_x.v(buf=1)[:, :] = a.v(buf=1) - 0.5*(1.0 + cx.v(buf=1))*ldelta_ax.v(buf=1)
    ur_y.v(buf=1)[:, :] = a.v(buf=1) - 0.5*(1.0 + cy.v(buf=1))*ldelta_ay.v(buf=1)

    # Solve Riemann's problem to get the correct transverse term

    u_xt = riemann(my_data, ul_x, ur_x)
    u_yt = riemann(my_data, ul_y, ur_y)

    # # Compute the transverse correction flux based off from predictor term.

    F_xt = myg.scratch_array()
    F_yt = myg.scratch_array()

    if scalar_name == "x-velocity":
        F_xt.v(buf=1)[:, :] = 0.5 * u_xt.v(buf=1) * u_xt.v(buf=1)
        F_yt.v(buf=1)[:, :] = v.v(buf=1) * u_yt.v(buf=1)

        # F_xt.v(buf=1)[:, :] = u_xt.v(buf=1) * u_xt.v(buf=1)
        # F_yt.v(buf=1)[:, :] = v.v(buf=1) * u_yt.v(buf=1)

        # F_xt.v(buf=1)[:, :] = u.v(buf=1) * u_xt.v(buf=1)
        # F_yt.v(buf=1)[:, :] = v.v(buf=1) * u_yt.v(buf=1)

    elif scalar_name == "y-velocity":
        F_xt.v(buf=1)[:, :] = u.v(buf=1) * u_xt.v(buf=1)
        F_yt.v(buf=1)[:, :] = 0.5 * u_yt.v(buf=1) * u_yt.v(buf=1)

        # F_xt.v(buf=1)[:, :] = u.v(buf=1) * u_xt.v(buf=1)
        # F_yt.v(buf=1)[:, :] = u_yt.v(buf=1) * u_yt.v(buf=1)

        # F_xt.v(buf=1)[:, :] = u.v(buf=1) * u_xt.v(buf=1)
        # F_yt.v(buf=1)[:, :] = v.v(buf=1) * u_yt.v(buf=1)

    else:
        F_xt.v(buf=1)[:, :] = u.v(buf=1) * u_xt.v(buf=1)
        F_yt.v(buf=1)[:, :] = v.v(buf=1) * u_yt.v(buf=1)

    # Apply transverse correction terms:

    ul_x.v(buf=1)[:, :] = ul_x.v(buf=1) - 0.5*dt/myg.dy*(F_yt.ip_jp(-1, 1, buf=1) - F_yt.ip(-1, buf=1))
    ul_y.v(buf=1)[:, :] = ul_y.v(buf=1) - 0.5*dt/myg.dx*(F_xt.ip_jp(1, -1, buf=1) - F_xt.jp(-1, buf=1))
    ur_x.v(buf=1)[:, :] = ur_x.v(buf=1) - 0.5*dt/myg.dy*(F_yt.ip_jp(0, 1, buf=1) - F_yt.ip_jp(0, 0, buf=1))
    ur_y.v(buf=1)[:, :] = ur_y.v(buf=1) - 0.5*dt/myg.dx*(F_xt.ip_jp(1, 0, buf=1) - F_xt.ip_jp(0, 0, buf=1))

    # solve for riemann's problem again

    u_x = riemann(my_data, ul_x, ur_x)
    u_y = riemann(my_data, ul_y, ur_y)

    # Compute the actual flux.

    F_x = myg.scratch_array()
    F_y = myg.scratch_array()

    if scalar_name == "x-velocity":

        F_x.v(buf=1)[:, :] = 0.5 * u_x.v(buf=1)*u_x.v(buf=1)
        F_y.v(buf=1)[:, :] = v.v(buf=1)*u_y.v(buf=1)

    elif scalar_name == "y-velocity":

        F_x.v(buf=1)[:, :] = u.v(buf=1)*u_x.v(buf=1)
        F_y.v(buf=1)[:, :] = 0.5 * u_y.v(buf=1)*u_y.v(buf=1)

    else:

        F_x.v(buf=1)[:, :] = u.v(buf=1)*u_x.v(buf=1)
        F_y.v(buf=1)[:, :] = v.v(buf=1)*u_y.v(buf=1)

    return F_x, F_y
