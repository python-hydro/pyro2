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

            b = (0, 1)

            eta_x.v(buf=b)[:, :] = 0.5*(eta.ip(-1, buf=b) + eta.v(buf=b))
            eta_y.v(buf=b)[:, :] = 0.5*(eta.jp(-1, buf=b) + eta.v(buf=b))

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

        b = (0, 1, 0, 0)
        c_eta_x.v(buf=b)[:, :] = 0.5*(self.x.v(buf=b, s=2) + self.x.jp(1, buf=b, s=2))

        b = (0, 0, 0, 1)
        c_eta_y.v(buf=b)[:, :] = 0.5*(self.y.v(buf=b, s=2) + self.y.ip(1, buf=b, s=2))

        # redo the normalization
        c_edge_coeffs.x = c_eta_x*fg.dx**2/cg.dx**2
        c_edge_coeffs.y = c_eta_y*fg.dy**2/cg.dy**2

        return c_edge_coeffs
