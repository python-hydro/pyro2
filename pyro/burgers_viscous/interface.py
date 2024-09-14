from pyro.multigrid import MG


def get_lap(grid, a):
    r"""
    Parameters
    ----------
    grid: grid object
    a: ndarray
       the variable that we want to find the laplacian of

    Returns
    -------
    out : ndarray (laplacian of state a)
    """

    lap = grid.scratch_array()

    lap.v(buf=2)[:, :] = (a.ip(1, buf=2) - 2.0*a.v(buf=2) + a.ip(-1, buf=2)) / grid.dx**2 \
                       + (a.jp(1, buf=2) - 2.0*a.v(buf=2) + a.jp(-1, buf=2)) / grid.dy**2

    return lap


def diffuse(my_data, rp, dt, scalar_name, A):
    r"""
    A routine to solve the Helmhotlz equation with constant coefficient
    and update the state.

    (a + b \lap) phi = f

    using Crank-Nicolson discretization with multigrid V-cycle.

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
    A: ndarray
       The advective source term for diffusion

    Returns
    -------
    out : ndarray (solution of the Helmholtz equation)

    """

    myg = my_data.grid

    a = my_data.get_var(scalar_name)
    eps = rp.get_param("diffusion.eps")

    # Create the multigrid with a constant diffusion coefficient
    # the equation has form:
    # (alpha - beta L) phi = f
    #
    # with alpha = 1
    #      beta  = (dt/2) k
    #      f     = phi + (dt/2) k L phi - A
    #
    # Same as Crank-Nicolson discretization except with an extra
    # advection source term)

    mg = MG.CellCenterMG2d(myg.nx, myg.ny,
                           xmin=myg.xmin, xmax=myg.xmax,
                           ymin=myg.ymin, ymax=myg.ymax,
                           xl_BC_type=my_data.BCs[scalar_name].xlb,
                           xr_BC_type=my_data.BCs[scalar_name].xrb,
                           yl_BC_type=my_data.BCs[scalar_name].ylb,
                           yr_BC_type=my_data.BCs[scalar_name].yrb,
                           alpha=1.0, beta=0.5*dt*eps,
                           verbose=0)

    # Compute the RHS: f

    f = mg.soln_grid.scratch_array()

    lap = get_lap(myg, a)

    f.v()[:, :] = a.v() + 0.5*dt*eps * lap.v() - dt*A.v()

    mg.init_RHS(f)

    # initial guess is zeros

    mg.init_zeros()
    mg.solve(rtol=1.e-12)

    # perform the diffusion update

    a.v()[:, :] = mg.get_solution().v()


def apply_diffusion_corrections(grid, dt, eps,
                                u, v,
                                u_xl, u_xr,
                                u_yl, u_yr,
                                v_xl, v_xr,
                                v_yl, v_yr):
    r"""
    Apply diffusion correction term to the interface state

    .. math::

       u_t  + u u_x  + v u_y  = eps (u_xx + u_yy)
       v_t  + u v_x  + v v_y  = eps (v_xx + v_yy)

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
    grid : Grid2d
        The grid object
    dt : float
        The timestep
    eps : float
         The viscosity
    u_xl, u_xr : ndarray ndarray
        left and right states of x-velocity in x-interface.
    u_yl, u_yr : ndarray ndarray
        left and right states of x-velocity in y-interface.
    v_xl, v_xr : ndarray ndarray
        left and right states of y-velocity in x-interface.
    v_yl, u_yr : ndarray ndarray
        left and right states of y-velocity in y-interface.

    Returns
    -------
    out : ndarray, ndarray, ndarray, ndarray, ndarray, ndarray, ndarray, ndarray
        unsplit predictions of the left and right states of u and v on both the x- and
        y-interfaces along with diffusion correction terms.
    """

    #apply diffusion correction to the interface

    lap_u = get_lap(grid, u)
    lap_v = get_lap(grid, v)

    u_xl.ip(1, buf=2)[:, :] += 0.5 * eps * dt * lap_u.v(buf=2)
    u_yl.jp(1, buf=2)[:, :] += 0.5 * eps * dt * lap_u.v(buf=2)
    u_xr.v(buf=2)[:, :] += 0.5 * eps * dt * lap_u.v(buf=2)
    u_yr.v(buf=2)[:, :] += 0.5 * eps * dt * lap_u.v(buf=2)

    v_xl.ip(1, buf=2)[:, :] += 0.5 * eps * dt * lap_v.v(buf=2)
    v_yl.jp(1, buf=2)[:, :] += 0.5 * eps * dt * lap_v.v(buf=2)
    v_xr.v(buf=2)[:, :] += 0.5 * eps * dt * lap_v.v(buf=2)
    v_yr.v(buf=2)[:, :] += 0.5 * eps * dt * lap_v.v(buf=2)

    return u_xl, u_xr, u_yl, u_yr, v_xl, v_xr, v_yl, v_yr
