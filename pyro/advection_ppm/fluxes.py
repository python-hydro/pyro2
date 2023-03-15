import numpy as np
import pyro.mesh.patch as patch
from pyro.mesh.reconstruction import ppm_reconstruction
import pyro.mesh.array_indexer as ai


def riemman(myg, u, v, ar, al, idim):
    """Here we write the riemmann problem for the advection equation. The solution
     is trivial: from ql and qr, we choose the upwind case."""

    #create an array with all the discontinuities
    r = myg.scratch_array()

    if idim == 1:
        if u > 0:
            r.v(buf=2)[:, :] = al.v(buf=2)
        else:
            r.v(buf=2)[:, :] = ar.ip(1, buf=2)
    elif idim == 2:
        if v > 0:
            r.v(buf=2)[:, :] = al.v(buf=2)
        else:
            r.v(buf=2)[:, :] = ar.jp(1, buf=2)

    return r


def ctu_unsplit_fluxes(data, rp, dt, scalar_name):
    """
    From here we plan to unsplit the fluxes according to the CTU method, but using a PPM reconstruction.
    In order to compute the fluxes states a_{i+1/2,j}^{n+1/2} we:

    1. Compute all the normal states for each interface. store them in a_xr[:] amd a_xl[:].

    """

    #We start by calling our data first.
    myg = data.grid
    a = data.get_var(scalar_name)

    #Now,let's call the velocity parameters
    #and compute the cfl number of each dimesnion
    u = rp.get_param("advection.u")
    v = rp.get_param("advection.v")

    cx = u*dt / myg.dx
    cy = v*dt / myg.dy

    #Now, using ar and al, we have to construct the normal states on each interface limit.

    #Let us start with idir == 1:
    delta_ax = myg.scratch_array()
    a6x = myg.scratch_array()

    _ar, _al = ppm_reconstruction(a, myg, idir=1)
    ar = ai.ArrayIndexer(_ar, myg)
    al = ai.ArrayIndexer(_al, myg)

    delta_ax.v(buf=1)[:, :] = ar.v(buf=1) - al.v(buf=1)
    a6x.v(buf=1)[:, :] = 6.0 * (a.v(buf=1) - 0.5*(ar.v(buf=1) + al.v(buf=1)))

    ax_normal_l = myg.scratch_array()
    ax_normal_r = myg.scratch_array()

    ax_normal_l.v(buf=1)[:, :] = al.v(buf=1) - cx * (delta_ax.v(buf=1) - (1 - 2.0 * cx / 3.0) * a6x.v(buf=1))
    ax_normal_r.v(buf=1)[:, :] = ar.ip(1, buf=1) + cx * (delta_ax.ip(1, buf=1) + (1 - 2*cx/3.0) * a6x.ip(1,buf=1))

    #Let us now move to idir==2:
    delta_ay = myg.scratch_array()
    a6y = myg.scratch_array()

    _ar, _al = ppm_reconstruction(a, myg, idir=2)
    ar = ai.ArrayIndexer(_ar, myg)
    al = ai.ArrayIndexer(_ar, myg)

    delta_ay.v(buf=1)[:, :] = ar.v(buf=1) - al.v(buf=1)
    a6y.v(buf=1)[:, :] = 6.0 * (a.v(buf=1) - 0.5*(ar.v(buf=1) + al.v(buf=1)))

    ay_normal_l = myg.scratch_array()
    ay_normal_r = myg.scratch_array()

    ay_normal_l.v(buf=1)[:, :] = al.v(buf=1) - cx * (delta_ay.v(buf=1) - (1 - 2.0 * cx / 3.0)*a6y.v(buf=1))
    ay_normal_r.v(buf=1)[:, :] = ar.jp(1, buf=1) + cx * (delta_ay.jp(1, buf=1) + (1 - 2*cx/3.0) * a6y.jp(1,buf=1))

    #Now we compute the Riemman Problem between to states, in order to compute the transverse states
    ax_T = myg.scratch_array()
    ay_T = myg.scratch_array()

    ax_T = riemman(myg, u, v, ax_normal_l, ax_normal_r, idim=1)
    ay_T = riemman(myg, u, v, ay_normal_l, ay_normal_r, idim=2)

    #Let's move to performing the flux tranverse corrections
    ax_l = myg.scratch_array()
    ax_r = myg.scratch_array()
    ay_l = myg.scratch_array()
    ay_r = myg.scratch_array()

    ax_l.v(buf=1)[:, :] = ax_normal_l.v(buf=1) - 0.5 * cy * (ay_T.v(buf=1) - ay_T.jp(-1, buf=1))
    ax_r.v(buf=1)[:, :] = ax_normal_r.v(buf=1) - 0.5 * cy * (ay_T.ip(1, buf=1) - ay_T.ip_jp(1, -1, buf=1))

    ay_l.v(buf=1)[:, :] = ay_normal_l.v(buf=1) - 0.5 * cx * (ax_T.v(buf=1) - ax_T.ip(-1, buf=1))
    ay_r.v(buf=1)[:, :] = ay_normal_r.v(buf=1) - 0.5 * cx * (ax_T.jp(1, buf=1) - ay_T.ip_jp(-1, 1 ,buf=1))

    #Finally we may perform another sequence of riemman problems
    ax = riemman(myg, u, v, ax_l, ax_r, idim=1)
    ay = riemman(myg, u, v,  ay_l, ay_r, idim=2)

    #From here we may compute the fluxes terms.
    Fx = myg.scratch_array()
    Fy = myg.scratch_array()

    Fx.v(buf=1)[:, :] = u*ax.v(buf=1)
    Fy.v(buf=1)[:, :] = v*ay.v(buf=1)

    return Fx, Fy