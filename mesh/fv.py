import mesh.patch as patch

class FV(patch.CellCenterData2d):
    """this is a finite-volume grid.  We expect the data to represent
    cell-averages, and do operations to 4th order.  This assumes dx = dy """

    def to_centers(self, name):
        """ convert variable name from an average to cell-centers """

        a = self.get_var(name)
        c = a.grid.scratch_array()

        c.v()[:,:] = a.v() - a.grid.dx**2*a.lap()/24.0
        return c


    def from_centers(self, name, c):
        """initialize variable name from cell-center data, converting to
        averages"""
        a = self.get_var(name)
        a.v()[:,:] = a.v() + a.grid.dx**2*a.lap()/24.0


