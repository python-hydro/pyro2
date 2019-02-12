"""
A generic Runge-Kutta type integrator for integrating CellCenterData2d.
We support a generic Butcher tableau for explicit the Runge-Kutta update::

   0   |
   c_2 | a_21
   c_3 | a_31 a_32
   :   |  :        .
   :   |  :          .
   c_s | a_s1 a_s2 ... a_s,s-1
   ----+---------------------------
       | b_1  b_2  ... b_{s-1}  b_s


the update is::

   y_{n+1} = y_n + dt sum_{i=1}^s {b_i k_i}

and the s increment is::

   k_s = f(t + c_s dt, y_n + dt (a_s1 k1 + a_s2 k2 + ... + a_s,s-1 k_{s-1})

"""

import numpy as np
import mesh.patch as patch
from copy import deepcopy

a = {}
b = {}
c = {}

nstages = {}

# second-order standard
a["RK2"] = np.array([[0.0, 0.0],
                     [0.5, 0.0]])

b["RK2"] = np.array([0.0, 1.0])

c["RK2"] = np.array([0.0, 0.5])


# second-order TVD (Gottlieb & Shu)
a["TVD2"] = np.array([[0.0, 0.0],
                      [1.0, 0.0]])

b["TVD2"] = np.array([0.5, 0.5])

c["TVD2"] = np.array([0.0, 1.0])


# third-order TVD (Gottlieb & Shu)
a["TVD3"] = np.array([[0.0,  0.0,  0.0],
                      [1.0,  0.0,  0.0],
                      [0.25, 0.25, 0.0]])

b["TVD3"] = np.array([1./6., 1./6., 2./3.])

c["TVD3"] = np.array([0.0, 1.0, 0.5])


# fourth-order
a["RK4"] = np.array([[0.0, 0.0, 0.0, 0.0],
                     [0.5, 0.0, 0.0, 0.0],
                     [0.0, 0.5, 0.0, 0.0],
                     [0.0, 0.0, 1.0, 0.0]])

b["RK4"] = np.array([1./6., 1./3., 1./3., 1./6.])

c["RK4"] = np.array([0.0, 0.5, 0.5, 1.0])


class RKIntegrator(object):
    """the integration class for CellCenterData2d, supporting RK
    integration"""

    def __init__(self, t, dt, method="RK4"):
        """t is the starting time, dt is the total timestep to advance, method
        = {2,4} is the temporal method"""
        self.method = method

        self.t = t
        self.dt = dt

        # storage for the intermediate stages
        self.k = [None]*len(b[self.method])

        self.start = None

    def nstages(self):
        """return the number of stages"""
        return len(b[self.method])

    def set_start(self, start):
        """store the starting conditions (should be a CellCenterData2d
        object)"""
        self.start = start

    def store_increment(self, istage, k_stage):
        """store the increment for stage istage -- this should not have a dt
        weighting"""
        self.k[istage] = k_stage

    def get_stage_start(self, istage):
        """get the starting conditions (a CellCenterData2d object) for stage
        istage"""
        if istage == 0:
            ytmp = self.start
        else:
            ytmp = patch.cell_center_data_clone(self.start)
            for n in range(ytmp.nvar):
                var = ytmp.get_var_by_index(n)
                for s in range(istage):
                    var.v()[:, :] += self.dt*a[self.method][istage, s]*self.k[s].v(n=n)[:, :]

            ytmp.t = self.t + c[self.method][istage]*self.dt

        return ytmp

    def compute_final_update(self):
        """this constructs the final t + dt update, overwriting the inital data"""
        ytmp = self.start
        for n in range(ytmp.nvar):
            var = ytmp.get_var_by_index(n)
            for s in range(self.nstages()):
                var.v()[:, :] += self.dt*b[self.method][s]*self.k[s].v(n=n)[:, :]

        return ytmp

    def __str__(self):
        return "integration method: {}; number of stages: {}".format(self.method, self.nstages())


class RKIntegratorMHD(RKIntegrator):
    """the integration class for CellCenterData2d, supporting RK
    integration"""

    def __init__(self, t, dt, method="RK4"):
        """t is the starting time, dt is the total timestep to advance, method
        = {2,4} is the temporal method"""
        super(RKIntegratorMHD, self).__init__(t, dt, method=method)

        self.kx = [None]*len(b[self.method])
        self.ky = [None]*len(b[self.method])


    def store_increment(self, istage, k_stage, kx_stage, ky_stage):
        """store the increment for stage istage -- this should not have a dt
        weighting"""
        self.k[istage] = k_stage
        self.kx[istage] = kx_stage
        self.ky[istage] = ky_stage

    def get_stage_start(self, istage):
        """get the starting conditions (a CellCenterData2d object) for stage
        istage"""
        if istage == 0:
            ytmp, fxtmp, fytmp = self.start
        else:
            ytmp, fxtmp, fytmp = self.start
            ytmp = patch.cell_center_data_clone(ytmp)
            fxtmp = deepcopy(fxtmp)
            fytmp = deepcopy(fytmp)

            for n in range(ytmp.nvar):
                var = ytmp.get_var_by_index(n)
                for s in range(istage):
                    var.v()[:, :] += self.dt*a[self.method][istage, s]*self.k[s].v(n=n)[:, :]

            bx = fxtmp.get_var("x-magnetic-field")
            by = fytmp.get_var("y-magnetic-field")
            for s in range(istage):
                bx.v()[:,:] += self.dt*a[self.method][istage, s]*self.kx[s].v()[:, :]
                by.v()[:,:] += self.dt*a[self.method][istage, s]*self.ky[s].v()[:, :]

            ytmp.get_var("x-magnetic-field").v()[:, :] = 0.5 * (bx.ip(1)[:-1, :] + bx.v()[:-1, :])
            ytmp.get_var("y-magnetic-field").v()[:, :] = 0.5 * (by.jp(1)[:, :-1] + by.v()[:, :-1])

            ytmp.t = self.t + c[self.method][istage]*self.dt

        return ytmp, fxtmp, fytmp

    def compute_final_update(self):
        """this constructs the final t + dt update, overwriting the inital data"""
        ytmp, fxtmp, fytmp = self.start
        for n in range(ytmp.nvar):
            var = ytmp.get_var_by_index(n)
            for s in range(self.nstages()):
                var.v()[:, :] += self.dt*b[self.method][s]*self.k[s].v(n=n)[:, :]

        bx = fxtmp.get_var("x-magnetic-field")
        by = fytmp.get_var("y-magnetic-field")

        for s in range(self.nstages()):
            bx.v()[:, :] += self.dt*b[self.method][s]*self.kx[s].v()[:, :]
            by.v()[:, :] += self.dt*b[self.method][s]*self.ky[s].v()[:, :]

        ytmp.get_var("x-magnetic-field").v()[:, :] = 0.5 * (bx.ip(1)[:-1, :] + bx.v()[:-1, :])
        ytmp.get_var("y-magnetic-field").v()[:, :] = 0.5 * (by.jp(1)[:, :-1] + by.v()[:, :-1])

        return ytmp, fxtmp, fytmp

    def __str__(self):
        return "integration method: {}; number of stages: {}".format(self.method, self.nstages())
