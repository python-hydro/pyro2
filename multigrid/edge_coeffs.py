class EdgeCoeffs:
    """
    a simple container class to hold edge-centered coefficients
    and restrict them to coarse levels
    """
    def __init__(self, g, eta, empty=False):

        self.grid = g

        if not empty:
            eta_x = g.scratch_array()
            eta_y = g.scratch_array()

            # the eta's are defined on the interfaces, so
            # eta_x[i,j] will be eta_{i-1/2,j} and
            # eta_y[i,j] will be eta_{i,j-1/2}

            eta_x[g.ilo:g.ihi+2,g.jlo:g.jhi+1] = \
                0.5*(eta[g.ilo-1:g.ihi+1,g.jlo:g.jhi+1] +
                     eta[g.ilo  :g.ihi+2,g.jlo:g.jhi+1])

            eta_y[g.ilo:g.ihi+1,g.jlo:g.jhi+2] = \
                0.5*(eta[g.ilo:g.ihi+1,g.jlo-1:g.jhi+1] +
                     eta[g.ilo:g.ihi+1,g.jlo  :g.jhi+2])

            eta_x /= g.dx**2
            eta_y /= g.dy**2

            self.x = eta_x
            self.y = eta_y


    def restrict(self):
        """
        restrict the edge values to a coarser grid.  Return a new
        EdgeCoeffs object
        """

        cg = self.grid.coarse_like(2)

        c_edge_coeffs = EdgeCoeffs(cg, None, empty=True)

        c_eta_x = cg.scratch_array()
        c_eta_y = cg.scratch_array()

        fg = self.grid

        c_eta_x[cg.ilo:cg.ihi+2,cg.jlo:cg.jhi+1] = \
            0.5*(self.x[fg.ilo:fg.ihi+2:2,fg.jlo  :fg.jhi+1:2] +
                 self.x[fg.ilo:fg.ihi+2:2,fg.jlo+1:fg.jhi+1:2])

        # redo the normalization
        c_edge_coeffs.x = c_eta_x*fg.dx**2/cg.dx**2

        c_eta_y[cg.ilo:cg.ihi+1,cg.jlo:cg.jhi+2] = \
            0.5*(self.y[fg.ilo  :fg.ihi+1:2,fg.jlo:fg.jhi+2:2] +
                 self.y[fg.ilo+1:fg.ihi+1:2,fg.jlo:fg.jhi+2:2])

        c_edge_coeffs.y = c_eta_y*fg.dy**2/cg.dy**2

        return c_edge_coeffs


