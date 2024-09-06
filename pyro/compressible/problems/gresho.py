import numpy as np

from pyro.util import msg

DEFAULT_INPUTS = "inputs.gresho"

PROBLEM_PARAMS = {"gresho.rho0": 1.0,  # density in the domain
                  "gresho.r": 0.2,  # radial location of peak velocity
                  "gresho.p0": 59.5,  # ambient pressure in the domain
                  "gresho.t_r": 1.0}  # reference time (used for setting peak velocity)


def init_data(my_data, rp):
    """ initialize the Gresho vortex problem """

    if rp.get_param("driver.verbose"):
        msg.bold("initializing the Gresho vortex problem...")

    # get the density and velocities
    dens = my_data.get_var("density")
    xmom = my_data.get_var("x-momentum")
    ymom = my_data.get_var("y-momentum")
    ener = my_data.get_var("energy")

    myg = my_data.grid

    x_center = 0.5 * (myg.x[0] + myg.x[-1])
    y_center = 0.5 * (myg.y[0] + myg.y[-1])
    L_x = myg.xmax - myg.xmin

    gamma = rp.get_param("eos.gamma")

    rho0 = rp.get_param("gresho.rho0")
    p0 = rp.get_param("gresho.p0")

    rr = rp.get_param("gresho.r")
    t_r = rp.get_param("gresho.t_r")

    q_r = 0.4 * np.pi * L_x / t_r

    pres = myg.scratch_array()
    u_phi = myg.scratch_array()

    dens[:, :] = rho0
    pres[:, :] = p0

    rad = np.sqrt((myg.x2d - x_center)**2 +
                  (myg.y2d - y_center)**2)

    indx1 = rad < rr
    u_phi[indx1] = 5.0 * rad[indx1]
    pres[indx1] = p0 + 12.5 * rad[indx1]**2

    indx2 = np.logical_and(rad >= rr, rad < 2.0*rr)
    u_phi[indx2] = 2.0 - 5.0 * rad[indx2]
    pres[indx2] = p0 + 12.5 * rad[indx2]**2 + \
        4.0 * (1.0 - 5.0 * rad[indx2] - np.log(rr) + np.log(rad[indx2]))

    indx3 = rad >= 2.0 * rr
    u_phi[indx3] = 0.0
    pres[indx3] = p0 + 12.5 * (2.0 * rr)**2 + \
        4.0 * (1.0 - 5.0 * (2.0 * rr) - np.log(rr) + np.log(2.0 * rr))

    xmom[:, :] = -dens[:, :] * q_r * u_phi[:, :] * (myg.y2d - y_center) / rad[:, :]
    ymom[:, :] = dens[:, :] * q_r * u_phi[:, :] * (myg.x2d - x_center) / rad[:, :]

    ener[:, :] = pres[:, :] / (gamma - 1.0) + \
                0.5 * (xmom[:, :]**2 + ymom[:, :]**2) / dens[:, :]

    # report peak Mach number
    cs = np.sqrt(gamma * pres / dens)
    M = np.abs(q_r * u_phi).max() / cs.max()
    print(f"peak Mach number = {M}")


def finalize():
    """ print out any information to the userad at the end of the run """
