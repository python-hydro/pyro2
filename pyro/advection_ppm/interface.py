from pyro.mesh.reconstruction import ppm_reconstruction


def ppm_interface(a, myg, rp, dt):

    # get the advection velocities
    u = rp.get_param("advection.u")
    v = rp.get_param("advection.v")

    limiter = rp.get_param("advection.limiter")

    cx = u*dt/myg.dx
    cy = v*dt/myg.dy

    # --------------------------------------------------------------------------
    # monotonized central differences
    # --------------------------------------------------------------------------

    delta_ax = myg.scratch_array()
    a6x = myg.scratch_array()
    delta_ay = myg.scratch_array()
    a6y = myg.scratch_array()

    # upwind
    a_x = myg.scratch_array()

    ar, al = ppm_reconstruction(a, myg, idir=1, limiter=limiter)
    delta_ax.v(buf=1)[:, :] = al.v(buf=1) - ar.v(buf=1)
    a6x.v(buf=1)[:, :] = 6.0 * (a.v(buf=1) - 0.5*(ar.v(buf=1) + al.v(buf=1)))

    if u < 0:
        cx = -cx
        # a_x[i,j] = a[i,j] - 0.5*(1.0 + cx)*ldelta_a[i,j]
        a_x.v(buf=1)[:, :] = ar.v(buf=1) + 0.5 * cx * (delta_ax.v(buf=1) +
                                (1 - 2.0 * cx / 3.0) * a6x.v(buf=1))
    else:
        # a_x[i,j] = a[i-1,j] + 0.5*(1.0 - cx)*ldelta_a[i-1,j]
        a_x.v(buf=1)[:, :] = al.ip(-1, buf=1) - 0.5 * cx * (delta_ax.ip(-1, buf=1) -
                                (1 - 2.0 * cx / 3.0) * a6x.ip(-1, buf=1))

    # y-direction
    a_y = myg.scratch_array()

    ar, al = ppm_reconstruction(a, myg, idir=2, limiter=limiter)
    delta_ay.v(buf=1)[:, :] = al.v(buf=1) - ar.v(buf=1)
    a6y.v(buf=1)[:, :] = 6.0 * (a.v(buf=1) - 0.5*(ar.v(buf=1) + al.v(buf=1)))

    # upwind
    if v < 0:
        cy = -cy
        # a_y[i,j] = a[i,j] - 0.5*(1.0 + cy)*ldelta_a[i,j]
        a_y.v(buf=1)[:, :] = ar.v(buf=1) + 0.5 * cy * (delta_ay.v(buf=1) +
                                (1 - 2.0 * cy / 3.0) * a6y.v(buf=1))
    else:
        # a_y[i,j] = a[i,j-1] + 0.5*(1.0 - cy)*ldelta_a[i,j-1]
        a_y.v(buf=1)[:, :] = al.jp(-1, buf=1) - 0.5 * cy * (delta_ay.jp(-1, buf=1) -
                                (1 - 2.0 * cy / 3.0) * a6y.jp(-1, buf=1))

    return u, v, a_x, a_y
