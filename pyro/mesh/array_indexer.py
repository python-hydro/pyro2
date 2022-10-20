"""An array class that has methods supporting the type of stencil
operations we see in finite-difference methods, like i+1, i-1, etc.

"""


import numpy as np


def _buf_split(b):
    """ take an integer or iterable and break it into a -x, +x, -y, +y
    value representing a ghost cell buffer
    """
    try:
        bxlo, bxhi, bylo, byhi = b
    except (ValueError, TypeError):
        try:
            blo, bhi = b
        except (ValueError, TypeError):
            blo = b
            bhi = b
        bxlo = bylo = blo
        bxhi = byhi = bhi
    return bxlo, bxhi, bylo, byhi


class ArrayIndexer(np.ndarray):
    """a class that wraps the data region of a single cell-centered data
        array (d) and allows us to easily do array operations like
        d[i+1,j] using the ip() method.

    """

    def __new__(cls, d, grid=None):
        obj = np.asarray(d).view(cls)
        obj.g = grid
        obj.c = len(d.shape)

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.g = getattr(obj, "g", None)
        self.c = getattr(obj, "c", None)

    def __array_wrap__(self, out_arr, context=None):
        return np.ndarray.__array_wrap__(self, out_arr, context)  # pylint: disable=E1121

    def v(self, buf=0, n=0, s=1):
        """return a view of the valid data region for component n, with stride
        s, and a buffer of ghost cells given by buf

        """
        return self.ip_jp(0, 0, buf=buf, n=n, s=s)

    def ip(self, shift, buf=0, n=0, s=1):
        """return a view of the data shifted by shift in the x direction.  By
        default the view is the same size as the valid region, but the
        buf can specify how many ghost cells on each side to include.
        The component is n and s is the stride

        """
        return self.ip_jp(shift, 0, buf=buf, n=n, s=s)

    def jp(self, shift, buf=0, n=0, s=1):
        """return a view of the data shifted by shift in the y direction.  By
        default the view is the same size as the valid region, but the
        buf can specify how many ghost cells on each side to include.
        The component is n and s is the stride

        """
        return self.ip_jp(0, shift, buf=buf, n=n, s=s)

    def ip_jp(self, ishift, jshift, buf=0, n=0, s=1):
        """return a view of the data shifted by ishift in the x direction and
        jshift in the y direction.  By default the view is the same
        size as the valid region, but the buf can specify how many
        ghost cells on each side to include.  The component is n and s
        is the stride

        """
        bxlo, bxhi, bylo, byhi = _buf_split(buf)
        c = len(self.shape)

        if c == 2:
            return np.asarray(self[self.g.ilo-bxlo+ishift:self.g.ihi+1+bxhi+ishift:s,
                                   self.g.jlo-bylo+jshift:self.g.jhi+1+byhi+jshift:s])

        return np.asarray(self[self.g.ilo-bxlo+ishift:self.g.ihi+1+bxhi+ishift:s,
                               self.g.jlo-bylo+jshift:self.g.jhi+1+byhi+jshift:s, n])

    def lap(self, n=0, buf=0):
        """return the 5-point Laplacian"""
        l = (self.ip(-1, n=n, buf=buf) - 2*self.v(n=n, buf=buf) + self.ip(1, n=n, buf=buf))/self.g.dx**2 + \
            (self.jp(-1, n=n, buf=buf) - 2*self.v(n=n, buf=buf) + self.jp(1, n=n, buf=buf))/self.g.dy**2
        return l

    def norm(self, n=0):
        """
        find the norm of the quantity (index n) defined on the same grid,
        in the domain's valid region

        """
        c = len(self.shape)
        if c == 2:
            return np.sqrt(self.g.dx * self.g.dy *
                           np.sum((self[self.g.ilo:self.g.ihi+1, self.g.jlo:self.g.jhi+1]**2).flat))

        _tmp = self[:, :, n]
        return np.sqrt(self.g.dx * self.g.dy *
                       np.sum((_tmp[self.g.ilo:self.g.ihi+1, self.g.jlo:self.g.jhi+1]**2).flat))

    def copy(self):
        """make a copy of the array, defined on the same grid"""
        return ArrayIndexer(np.asarray(self).copy(), grid=self.g)

    def is_symmetric(self, nodal=False, tol=1.e-14, asymmetric=False):
        """return True is the data is left-right symmetric (to the tolerance
        tol) For node-centered data, set nodal=True

        """

        # prefactor to convert from symmetric to asymmetric test
        s = 1
        if asymmetric:
            s = -1

        if not nodal:
            L = self[self.g.ilo:self.g.ilo+self.g.nx//2,
                     self.g.jlo:self.g.jhi+1]
            R = self[self.g.ilo+self.g.nx//2:self.g.ihi+1,
                     self.g.jlo:self.g.jhi+1]
        else:
            L = self[self.g.ilo:self.g.ilo+self.g.nx//2+1,
                     self.g.jlo:self.g.jhi+1]
            print(self.g.ilo+self.g.nx//2, self.g.ihi+2)
            R = self[self.g.ilo+self.g.nx//2:self.g.ihi+2,
                     self.g.jlo:self.g.jhi+1]

        e = abs(L - s*np.flipud(R)).max()
        return e < tol

    def is_asymmetric(self, nodal=False, tol=1.e-14):
        """return True is the data is left-right asymmetric (to the tolerance
        tol)---e.g, the sign flips. For node-centered data, set nodal=True

        """
        return self.is_symmetric(nodal=nodal, tol=tol, asymmetric=True)

    def fill_ghost(self, n=0, bc=None):
        """Fill the boundary conditions.  This operates on a single component,
        n. We do periodic, reflect-even, reflect-odd, and outflow

        We need a BC object to tell us what BC type on each boundary.
        """

        # there is only a single grid, so every boundary is on
        # a physical boundary (except if we are periodic)

        # Note: we piggy-back on outflow and reflect-odd for
        # Neumann and Dirichlet homogeneous BCs respectively, but
        # this only works for a single ghost cell

        # -x boundary
        if bc.xlb in ["outflow", "neumann"]:
            if bc.xl_value is None:
                for i in range(self.g.ilo):
                    self[i, :, n] = self[self.g.ilo, :, n]
            else:
                self[self.g.ilo-1, :, n] = \
                    self[self.g.ilo, :, n] - self.g.dx*bc.xl_value[:]

        elif bc.xlb == "reflect-even":
            for i in range(self.g.ilo):
                self[i, :, n] = self[2*self.g.ng-i-1, :, n]

        elif bc.xlb in ["reflect-odd", "dirichlet"]:
            if bc.xl_value is None:
                for i in range(self.g.ilo):
                    self[i, :, n] = -self[2*self.g.ng-i-1, :, n]
            else:
                self[self.g.ilo-1, :, n] = \
                    2*bc.xl_value[:] - self[self.g.ilo, :, n]

        elif bc.xlb == "periodic":
            for i in range(self.g.ilo):
                self[i, :, n] = self[self.g.ihi-self.g.ng+i+1, :, n]

        # +x boundary
        if bc.xrb in ["outflow", "neumann"]:
            if bc.xr_value is None:
                for i in range(self.g.ihi+1, self.g.nx+2*self.g.ng):
                    self[i, :, n] = self[self.g.ihi, :, n]
            else:
                self[self.g.ihi+1, :, n] = \
                    self[self.g.ihi, :, n] + self.g.dx*bc.xr_value[:]

        elif bc.xrb == "reflect-even":
            for i in range(self.g.ng):
                i_bnd = self.g.ihi+1+i
                i_src = self.g.ihi-i

                self[i_bnd, :, n] = self[i_src, :, n]

        elif bc.xrb in ["reflect-odd", "dirichlet"]:
            if bc.xr_value is None:
                for i in range(self.g.ng):
                    i_bnd = self.g.ihi+1+i
                    i_src = self.g.ihi-i

                    self[i_bnd, :, n] = -self[i_src, :, n]
            else:
                self[self.g.ihi+1, :, n] = \
                    2*bc.xr_value[:] - self[self.g.ihi, :, n]

        elif bc.xrb == "periodic":
            for i in range(self.g.ihi+1, 2*self.g.ng + self.g.nx):
                self[i, :, n] = self[i-self.g.ihi-1+self.g.ng, :, n]

        # -y boundary
        if bc.ylb in ["outflow", "neumann"]:
            if bc.yl_value is None:
                for j in range(self.g.jlo):
                    self[:, j, n] = self[:, self.g.jlo, n]
            else:
                self[:, self.g.jlo-1, n] = \
                    self[:, self.g.jlo, n] - self.g.dy*bc.yl_value[:]

        elif bc.ylb == "reflect-even":
            for j in range(self.g.jlo):
                self[:, j, n] = self[:, 2*self.g.ng-j-1, n]

        elif bc.ylb in ["reflect-odd", "dirichlet"]:
            if bc.yl_value is None:
                for j in range(self.g.jlo):
                    self[:, j, n] = -self[:, 2*self.g.ng-j-1, n]
            else:
                self[:, self.g.jlo-1, n] = \
                    2*bc.yl_value[:] - self[:, self.g.jlo, n]

        elif bc.ylb == "periodic":
            for j in range(self.g.jlo):
                self[:, j, n] = self[:, self.g.jhi-self.g.ng+j+1, n]

        # +y boundary
        if bc.yrb in ["outflow", "neumann"]:
            if bc.yr_value is None:
                for j in range(self.g.jhi+1, self.g.ny+2*self.g.ng):
                    self[:, j, n] = self[:, self.g.jhi, n]
            else:
                self[:, self.g.jhi+1, n] = \
                    self[:, self.g.jhi, n] + self.g.dy*bc.yr_value[:]

        elif bc.yrb == "reflect-even":
            for j in range(self.g.ng):
                j_bnd = self.g.jhi+1+j
                j_src = self.g.jhi-j

                self[:, j_bnd, n] = self[:, j_src, n]

        elif bc.yrb in ["reflect-odd", "dirichlet"]:
            if bc.yr_value is None:
                for j in range(self.g.ng):
                    j_bnd = self.g.jhi+1+j
                    j_src = self.g.jhi-j

                    self[:, j_bnd, n] = -self[:, j_src, n]
            else:
                self[:, self.g.jhi+1, n] = \
                    2*bc.yr_value[:] - self[:, self.g.jhi, n]

        elif bc.yrb == "periodic":
            for j in range(self.g.jhi+1, 2*self.g.ng + self.g.ny):
                self[:, j, n] = self[:, j-self.g.jhi-1+self.g.ng, n]

    def pretty_print(self, n=0, fmt=None, show_ghost=True):
        """
        Print out a small dataset to the screen with the ghost cells
        a different color, to make things stand out
        """

        if fmt is None:
            if self.dtype == np.int:
                fmt = "%4d"
            elif self.dtype == np.float64:
                fmt = "%10.5g"
            else:
                raise ValueError("ERROR: dtype not supported")

        # print j descending, so it looks like a grid (y increasing
        # with height)
        if show_ghost:
            ilo = 0
            ihi = self.g.qx-1
            jlo = 0
            jhi = self.g.qy-1
        else:
            ilo = self.g.ilo
            ihi = self.g.ihi
            jlo = self.g.jlo
            jhi = self.g.jhi

        for j in reversed(range(jlo, jhi+1)):
            for i in range(ilo, ihi+1):

                if (j < self.g.jlo or j > self.g.jhi or
                    i < self.g.ilo or i > self.g.ihi):
                    gc = 1
                else:
                    gc = 0

                if self.c == 2:
                    val = self[i, j]
                else:
                    try:
                        val = self[i, j, n]
                    except IndexError:
                        val = self[i, j]

                if gc:
                    print("\033[31m" + fmt % (val) + "\033[0m", end="")
                else:
                    print(fmt % (val), end="")

            print(" ")

        leg = """
         ^ y
         |
         +---> x
        """
        print(leg)


class ArrayIndexerFC(ArrayIndexer):
    """a class that wraps the data region of a single face-centered data
        array (d) and allows us to easily do array operations like
        d[i+1,j] using the ip() method.

    """

    def __new__(cls, d, idir, grid=None):
        obj = np.asarray(d).view(cls)
        obj.g = grid
        obj.idir = idir
        obj.c = len(d.shape)

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.g = getattr(obj, "g", None)
        self.idir = getattr(obj, "idir", None)
        self.c = getattr(obj, "c", None)

    def __array_wrap__(self, out_arr, context=None):
        return np.ndarray.__array_wrap__(self, out_arr, context)  # pylint: disable=E1121

    def ip_jp(self, ishift, jshift, buf=0, n=0, s=1):
        """return a view of the data shifted by ishift in the x direction and
        jshift in the y direction.  By default the view is the same
        size as the valid region, but the buf can specify how many
        ghost cells on each side to include.  The component is n and s
        is the stride

        """
        bxlo, bxhi, bylo, byhi = _buf_split(buf)
        c = len(self.shape)

        if self.idir == 1:
            # face-centered in x
            if c == 2:
                return np.asarray(self[self.g.ilo-bxlo+ishift:self.g.ihi+2+bxhi+ishift:s,
                                       self.g.jlo-bylo+jshift:self.g.jhi+1+byhi+jshift:s])
            return np.asarray(self[self.g.ilo-bxlo+ishift:self.g.ihi+2+bxhi+ishift:s,
                                   self.g.jlo-bylo+jshift:self.g.jhi+1+byhi+jshift:s, n])
        else:  # idir == 2
            # face-centered in y
            if c == 2:
                return np.asarray(self[self.g.ilo-bxlo+ishift:self.g.ihi+1+bxhi+ishift:s,
                                       self.g.jlo-bylo+jshift:self.g.jhi+2+byhi+jshift:s])
            return np.asarray(self[self.g.ilo-bxlo+ishift:self.g.ihi+1+bxhi+ishift:s,
                                   self.g.jlo-bylo+jshift:self.g.jhi+2+byhi+jshift:s, n])

    def lap(self, n=0, buf=0):
        raise NotImplementedError("lap not implemented for ArrayIndexerFC")

    def norm(self, n=0):
        """
        find the norm of the quantity (index n) defined on the same grid,
        in the domain's valid region

        """
        c = len(self.shape)
        if self.idir == 1:
            if c == 2:
                return np.sqrt(self.g.dx * self.g.dy *
                               np.sum((self[self.g.ilo:self.g.ihi+2, self.g.jlo:self.g.jhi+1]**2).flat))

            _tmp = self[:, :, n]
            return np.sqrt(self.g.dx * self.g.dy *
                           np.sum((_tmp[self.g.ilo:self.g.ihi+2, self.g.jlo:self.g.jhi+1]**2).flat))

        # idir == 2
        if c == 2:
            return np.sqrt(self.g.dx * self.g.dy *
                           np.sum((self[self.g.ilo:self.g.ihi+1, self.g.jlo:self.g.jhi+2]**2).flat))

        _tmp = self[:, :, n]
        return np.sqrt(self.g.dx * self.g.dy *
                       np.sum((_tmp[self.g.ilo:self.g.ihi+1, self.g.jlo:self.g.jhi+2]**2).flat))

    def copy(self):
        """make a copy of the array, defined on the same grid"""
        return ArrayIndexerFC(np.asarray(self).copy(), self.idir, grid=self.g)

    def is_symmetric(self, nodal=False, tol=1.e-14, asymmetric=False):
        """return True is the data is left-right symmetric (to the tolerance
        tol) For node-centered data, set nodal=True

        """
        raise NotImplementedError()

    def is_asymmetric(self, nodal=False, tol=1.e-14):
        """return True is the data is left-right asymmetric (to the tolerance
        tol)---e.g, the sign flips. For node-centered data, set nodal=True

        """
        raise NotImplementedError()

    def fill_ghost(self, n=0, bc=None):
        """Fill the boundary conditions.  This operates on a single component,
        n. We do periodic, reflect-even, reflect-odd, and outflow

        We need a BC object to tell us what BC type on each boundary.
        """

        # there is only a single grid, so every boundary is on
        # a physical boundary (except if we are periodic)

        # Note: we piggy-back on outflow and reflect-odd for
        # Neumann and Dirichlet homogeneous BCs respectively, but
        # this only works for a single ghost cell

        # -x boundary
        if bc.xlb in ["outflow", "neumann", "reflect-even", "reflect-odd", "dirichlet"]:
            raise NotImplementedError("boundary condition not implemented for -x")
        elif bc.xlb == "periodic":
            if self.idir == 1:
                # face-centered in x
                for i in range(self.g.ilo):
                    self[i, :, n] = self[self.g.ihi-self.g.ng+i+1, :, n]
            elif self.idir == 2:
                # face-centered in y
                for i in range(self.g.ilo):
                    self[i, :, n] = self[self.g.ihi-self.g.ng+i+1, :, n]

        # +x boundary
        if bc.xrb in ["outflow", "neumann", "reflect-even", "reflect-odd", "dirichlet"]:
            raise NotImplementedError("boundary condition not implemented for +x")
        elif bc.xrb == "periodic":
            if self.idir == 1:
                # face-centered in x
                for i in range(self.g.ihi+2, 2*self.g.ng + self.g.nx + 1):
                    self[i, :, n] = self[i-self.g.ihi-1+self.g.ng, :, n]
            elif self.idir == 2:
                # face-centered in y
                for i in range(self.g.ihi+1, 2*self.g.ng + self.g.nx):
                    self[i, :, n] = self[i-self.g.ihi-1+self.g.ng, :, n]

        # -y boundary
        if bc.ylb in ["outflow", "neumann", "reflect-even", "reflect-odd", "dirichlet"]:
            raise NotImplementedError("boundary condition not implemented for -y")
        elif bc.ylb == "periodic":
            if self.idir == 1:
                # face-centered in x
                for j in range(self.g.jlo):
                    self[:, j, n] = self[:, self.g.jhi-self.g.ng+j+1, n]
            elif self.idir == 2:
                # face-centered in y
                for j in range(self.g.jlo):
                    self[:, j, n] = self[:, self.g.jhi-self.g.ng+j+1, n]

        # +y boundary
        if bc.yrb in ["outflow", "neumann", "reflect-even", "reflect-odd", "dirichlet"]:
            raise NotImplementedError("boundary condition not implemented for +y")
        elif bc.yrb == "periodic":
            if self.idir == 1:
                # face-centered in x
                for j in range(self.g.jhi+1, 2*self.g.ng + self.g.ny):
                    self[:, j, n] = self[:, j-self.g.jhi-1+self.g.ng, n]
            elif self.idir == 2:
                for j in range(self.g.jhi+2, 2*self.g.ng + self.g.ny + 1):
                    self[:, j, n] = self[:, j-self.g.jhi-1+self.g.ng, n]

    def pretty_print(self, n=0, fmt=None, show_ghost=True):
        """
        Print out a small dataset to the screen with the ghost cells
        a different color, to make things stand out
        """

        if fmt is None:
            if self.dtype == np.int:
                fmt = "%4d"
            elif self.dtype == np.float64:
                fmt = "%10.5g"
            else:
                raise ValueError("ERROR: dtype not supported")

        # print j descending, so it looks like a grid (y increasing
        # with height)
        if show_ghost:
            if self.idir == 1:
                ilo = 0
                ihi = self.g.qx
                jlo = 0
                jhi = self.g.qy-1
            elif self.idir == 2:
                ilo = 0
                ihi = self.g.qx-1
                jlo = 0
                jhi = self.g.qy

        else:
            if self.idir == 1:
                ilo = self.g.ilo
                ihi = self.g.ihi+1
                jlo = self.g.jlo
                jhi = self.g.jhi
            elif self.idir == 2:
                ilo = self.g.ilo
                ihi = self.g.ihi
                jlo = self.g.jlo
                jhi = self.g.jhi+1

        for j in reversed(range(jlo, jhi+1)):
            for i in range(ilo, ihi+1):

                if self.idir == 1:
                    if (j < self.g.jlo or j > self.g.jhi or
                        i < self.g.ilo or i > self.g.ihi+1):
                        gc = 1
                    else:
                        gc = 0
                elif self.idir == 2:
                    if (j < self.g.jlo or j > self.g.jhi+1 or
                        i < self.g.ilo or i > self.g.ihi):
                        gc = 1
                    else:
                        gc = 0

                if self.c == 2:
                    val = self[i, j]
                else:
                    val = self[i, j, n]

                if gc:
                    print("\033[31m" + fmt % (val) + "\033[0m", end="")
                else:
                    print(fmt % (val), end="")

            print(" ")

        leg = """
         ^ y
         |
         +---> x
        """
        print(leg)
