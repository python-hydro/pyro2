from __future__ import print_function

import numpy as np

def _buf_split(b):
    """ take an integer or iterable and break it into a -x, +x, -y, +y
    value representing a ghost cell buffer
    """
    try: bxlo, bxhi, bylo, byhi = b
    except:
        try: blo, bhi = b
        except:
            blo = b
            bhi = b
        bxlo = bylo = blo
        bxhi = byhi = bhi
    return bxlo, bxhi, bylo, byhi


class ArrayIndexer(np.ndarray):
    """ a class that wraps the data region of a single array (d)
        and allows us to easily do array operations like d[i+1,j]
        using the ip() method. """

    def __new__(self, d, grid=None):
        obj = np.asarray(d).view(self)
        obj.g = grid
        obj.c = len(d.shape)

        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.g = getattr(obj, "g", None)
        self.c = getattr(obj, "c", None)

    def __array_wrap__(self, out_arr, context=None):
        return np.ndarray.__array_wrap__(self, out_arr, context)

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
        else:
            return np.asarray(self[self.g.ilo-bxlo+ishift:self.g.ihi+1+bxhi+ishift:s,
                                   self.g.jlo-bylo+jshift:self.g.jhi+1+byhi+jshift:s,n])

    def norm(self, n=0):
        """
        find the norm of the quantity (index n) defined on the same grid,
        in the domain's valid region

        """
        c = len(self.shape)
        if c == 2:
            return self.g.norm(self)
        else:
            return self.g.norm(self[:,:,n])


    def copy(self):
        """make a copy of the array, defined on the same grid"""
        return ArrayIndexer(np.asarray(self).copy(), grid=self.g)


    def is_symmetric(self, nodal=False, tol=1.e-14, asymmetric=False):
        """return True is the data is left-right symmetric (to the tolerance
        tol) For node-centered data, set nodal=True

        """

        # prefactor to convert from symmetric to asymmetric test
        s = 1
        if asymmetric: s = -1

        if not nodal:
            L = self[self.g.ilo:self.g.ilo+self.g.nx//2,
                     self.g.jlo:self.g.jhi+1]
            R = self[self.g.ilo+self.g.nx//2:self.g.ihi+1,
                     self.g.jlo:self.g.jhi+1]
        else:
            L = self[self.g.ilo:self.g.ilo+self.g.nx//2+1,
                     self.g.jlo:self.g.jhi+1]
            print(self.g.ilo+self.g.nx//2,self.g.ihi+2)
            R = self[self.g.ilo+self.g.nx//2:self.g.ihi+2,
                     self.g.jlo:self.g.jhi+1]

        e = abs(L - s*np.flipud(R)).max()
        return e < tol


    def is_asymmetric(self, nodal=False, tol=1.e-14):
        """return True is the data is left-right asymmetric (to the tolerance
        tol)---e.g, the sign flips. For node-centered data, set nodal=True

        """
        return self.is_symmetric(nodal=nodal, tol=tol, asymmetric=True)


    def pretty_print(self, fmt=None):
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
        for j in reversed(range(self.g.qy)):
            for i in range(self.g.qx):

                if (j < self.g.jlo or j > self.g.jhi or
                    i < self.g.ilo or i > self.g.ihi):
                    gc = 1
                else:
                    gc = 0

                if gc:
                    print("\033[31m" + fmt % (self[i,j]) + "\033[0m", end="")
                else:
                    print(fmt % (self[i,j]), end="")

            print(" ")

        leg = """
         ^ y
         |
         +---> x
        """
        print(leg)
