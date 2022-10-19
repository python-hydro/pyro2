"""The routines that implement the 4th-order compressible scheme,
using SDC time integration

"""


import pyro.compressible_fv4 as compressible_fv4
import pyro.mesh.patch as patch
from pyro.util import msg


class Simulation(compressible_fv4.Simulation):
    """Drive the 4th-order compressible solver with SDC time integration"""

    def sdc_integral(self, m_start, m_end, As):
        """Compute the integral over the sources from m to m+1 with a
        Simpson's rule"""

        integral = self.cc_data.grid.scratch_array(nvar=self.ivars.nvar)

        if m_start == 0 and m_end == 1:
            for n in range(self.ivars.nvar):
                integral.v(n=n)[:, :] = self.dt/24.0 * (5.0*As[0].v(n=n) + 8.0*As[1].v(n=n) - As[2].v(n=n))

        elif m_start == 1 and m_end == 2:
            for n in range(self.ivars.nvar):
                integral.v(n=n)[:, :] = self.dt/24.0 * (-As[0].v(n=n) + 8.0*As[1].v(n=n) + 5.0*As[2].v(n=n))

        else:
            msg.fail("invalid quadrature range")

        return integral

    def evolve(self):

        """
        Evolve the equations of compressible hydrodynamics through a
        timestep dt.
        """

        tm_evolve = self.tc.timer("evolve")
        tm_evolve.begin()

        myd = self.cc_data

        # we need the solution at 3 time points and at the old and
        # current iteration (except for m = 0 -- that doesn't change).

        # This copy will initialize the the solution at all time nodes
        # with the current (old) solution.
        U_kold = []
        U_kold.append(patch.cell_center_data_clone(self.cc_data))
        U_kold.append(patch.cell_center_data_clone(self.cc_data))
        U_kold.append(patch.cell_center_data_clone(self.cc_data))

        U_knew = []
        U_knew.append(U_kold[0])
        U_knew.append(patch.cell_center_data_clone(self.cc_data))
        U_knew.append(patch.cell_center_data_clone(self.cc_data))

        # loop over iterations
        for _ in range(1, 5):

            # we need the advective term at all time nodes at the old
            # iteration -- we'll compute this now
            A_kold = []
            for m in range(3):
                _tmp = self.substep(U_kold[m])
                A_kold.append(_tmp)

            # loop over the time nodes and update
            for m in range(2):

                # update m to m+1 for knew

                # compute A(U_m^{k+1})
                A_knew = self.substep(U_knew[m])

                # compute the integral over A at the old iteration
                integral = self.sdc_integral(m, m+1, A_kold)

                # and the final update
                for n in range(self.ivars.nvar):
                    U_knew[m+1].data.v(n=n)[:, :] = U_knew[m].data.v(n=n) + \
                        0.5*self.dt * (A_knew.v(n=n) - A_kold[m].v(n=n)) + integral.v(n=n)

                # fill ghost cells
                U_knew[m+1].fill_BC_all()

            # store the current iteration as the old iteration
            for m in range(1, 3):
                U_kold[m].data[:, :, :] = U_knew[m].data[:, :, :]

        # store the new solution
        self.cc_data.data[:, :, :] = U_knew[-1].data[:, :, :]

        if self.particles is not None:
            self.particles.update_particles(self.dt)

        # increment the time
        myd.t += self.dt
        self.n += 1

        tm_evolve.end()
