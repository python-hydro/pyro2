"""This implements support for 4th-order accurate finite-volume data
by adding support for converting between cell averages and centers.

"""

from pyro.mesh.patch import CellCenterData2d


class FV2d(CellCenterData2d):
    """this is a finite-volume grid.  We expect the data to represent
    cell-averages, and do operations to 4th order.  This assumes dx =
    dy

    """

    def to_centers(self, name):
        """ convert variable name from an average to cell-centers """

        a = self.get_var(name)
        c = self.grid.scratch_array()
        ng = self.grid.ng
        c[:, :] = a[:, :]
        c.v(buf=ng-1)[:, :] = a.v(buf=ng-1) - self.grid.dx**2*a.lap(buf=ng-1)/24.0
        return c

    def from_centers(self, name):
        """treat the stored data as if it lives at cell-centers and convert
        it to an average

        """
        self.fill_BC(name)
        a = self.get_var(name)
        a.v()[:, :] = a.v() + self.grid.dx**2*a.lap()/24.0
