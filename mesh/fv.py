import mesh.patch as patch

class FV2d(patch.CellCenterData2d):
    """this is a finite-volume grid.  We expect the data to represent
    cell-averages, and do operations to 4th order.  This assumes dx =
    dy

    """

    def to_centers(self, name):
        """ convert variable name from an average to cell-centers """

        a = self.get_var(name)
        c = a.grid.scratch_array()
        ng = a.grid.ng
        c.v(buf=ng-1)[:,:] = a.v(buf=ng-1) - a.grid.dx**2*a.lap(buf=ng-1)/24.0
        return c


    def from_centers(self, name, a, c):
        """initialize variable name from average and cell-center data,
        converting to averages

        """
        a_stored = self.get_var(name)
        a_stored.v()[:,:] = c.v() + a.grid.dx**2*a.lap()/24.0


