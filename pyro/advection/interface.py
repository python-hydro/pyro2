from pyro.mesh import reconstruction


def linear_interface(a, myg, rp, dt):

    # get the advection velocities
    u = rp.get_param("advection.u")
    v = rp.get_param("advection.v")

    # --------------------------------------------------------------------------
    # monotonized central differences
    # --------------------------------------------------------------------------

    limiter = rp.get_param("advection.limiter")

    ldelta_ax = reconstruction.limit(a, myg, 1, limiter)
    ldelta_ay = reconstruction.limit(a, myg, 2, limiter)

    ldelta_Fx = reconstruction.limit(u*a, myg, 1, limiter)
    ldelta_Fy = reconstruction.limit(v*a, myg, 2, limiter)

    # x-direction
    a_x = myg.scratch_array()

    # upwind
    if u < 0:
        # a_x[i,j] = a[i,j] - 0.5*ldelta_ax[i,j] - 0.5*dt/dx*ldelta_Fx[i,j]
        a_x.v(buf=1)[:, :] = a.v(buf=1) - 0.5*ldelta_ax.v(buf=1) \
                                        - 0.5*dt/myg.dx*ldelta_Fx.v(buf=1)

    else:
        # a_x[i,j] = a[i-1,j] + 0.5*ldelta_ax[i-1,j] - 0.5*dt/dx*ldelta_Fx[i-1,j]
        a_x.v(buf=1)[:, :] = a.ip(-1, buf=1) + 0.5*ldelta_ax.ip(-1, buf=1) \
                                             - 0.5*dt/myg.dx*ldelta_Fx.ip(-1, buf=1)

    # y-direction
    a_y = myg.scratch_array()

    # upwind
    if v < 0:
        # a_y[i,j] = a[i,j] - 0.5*ldelta_ay[i,j] - 0.5*dt/dy*ldelta_Fy[i,j]
        a_y.v(buf=1)[:, :] = a.v(buf=1) - 0.5*ldelta_ay.v(buf=1) \
                                        - 0.5*dt/myg.dy*ldelta_Fy.v(buf=1)
    else:
        # a_y[i,j] = a[i,j-1] + 0.5*ldelta_ay[i,j-1] - 0.5*dt/dy*ldelta_Fy[i,j-1]
        a_y.v(buf=1)[:, :] = a.jp(-1, buf=1) + 0.5*ldelta_ay.jp(-1, buf=1) \
                                        - 0.5*dt/myg.dy*ldelta_Fy.jp(-1, buf=1)

    return u, v, a_x, a_y
