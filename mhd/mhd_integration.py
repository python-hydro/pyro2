from mesh.integration import *

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
