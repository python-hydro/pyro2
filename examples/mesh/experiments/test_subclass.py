# experiments in subclassing the ArrayIndexer
#
# this all follows:
#
# https://docs.scipy.org/doc/numpy-1.10.1/user/basics.subclassing.html
#
# current issues: when we do ip_jp(), perhaps we should just return an
# ndarray at that point, since we are going to be inconsistent with the
# grid (the view will have a different size than the grid object)
#
# maybe we use np.asarray() here? but this may make a copy and do away
# with the fact that we are just a view into the same memory...
# perhaps instead set the grid to None?
#
# alternately, perhaps we can use .view(self)? or something like that
# to ensure that it is just a view and no copy is done.  Some experiment
# on this is in mesh-exmaples.ipynb


import numpy as np

import mesh.array_indexer as ai
import mesh.patch as patch
from util import msg

_buf_split = ai._buf_split


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
        if obj is None:
            return
        self.g = getattr(obj, "g", None)
        self.c = getattr(obj, "c", None)

    def __array_wrap__(self, out_arr, context=None):
        return np.ndarray.__array_wrap__(self, out_arr, context)

    def v(self, buf=0, n=0, s=1):
        return self.ip_jp(0, 0, buf=buf, n=n, s=s)

    def ip(self, shift, buf=0, n=0, s=1):
        return self.ip_jp(shift, 0, buf=buf, n=n, s=s)

    def jp(self, shift, buf=0, n=0, s=1):
        return self.ip_jp(0, shift, buf=buf, n=n, s=s)

    def ip_jp(self, ishift, jshift, buf=0, n=0, s=1):
        bxlo, bxhi, bylo, byhi = _buf_split(buf)

        if self.c == 2:
            return self[self.g.ilo - bxlo + ishift:self.g.ihi + 1 + bxhi + ishift:s,
                        self.g.jlo - bylo + jshift:self.g.jhi + 1 + byhi + jshift:s]
        else:
            return self[self.g.ilo - bxlo + ishift:self.g.ihi + 1 + bxhi + ishift:s,
                        self.g.jlo - bylo + jshift:self.g.jhi + 1 + byhi + jshift:s, n]

    def norm(self, n=0):
        """
        find the norm of the quantity (index n) defined on the same grid,
        in the domain's valid region

        """
        if self.c == 2:
            return self.g.norm(self)
        else:
            return self.g.norm(self[:, :, n])

    def copy(self):
        return ArrayIndexer(np.asarray(self).copy(), grid=self.g)

    def is_symmetric(self, nodal=False, tol=1.e-14):
        if not nodal:
            L = self[self.g.ilo:self.g.ilo + self.g.nx / 2,
                     self.g.jlo:self.g.jhi + 1]
            R = self[self.g.ilo + self.g.nx / 2:self.g.ihi + 1,
                     self.g.jlo:self.g.jhi + 1]
        else:
            print(self.g.ilo, self.g.ilo + self.g.nx / 2 + 1)
            L = self[self.g.ilo:self.g.ilo + self.g.nx / 2 + 1,
                     self.g.jlo:self.g.jhi + 1]
            print(self.g.ilo + self.g.nx / 2, self.g.ihi + 2)
            R = self[self.g.ilo + self.g.nx / 2:self.g.ihi + 2,
                     self.g.jlo:self.g.jhi + 1]

        e = abs(L - np.flipud(R)).max()
        print(e, tol, e < tol)
        return e < tol

    def is_asymmetric(self, nodal=False, tol=1.e-14):
        if not nodal:
            L = self[self.g.ilo:self.g.ilo + self.g.nx / 2,
                     self.g.jlo:self.g.jhi + 1]
            R = self[self.g.ilo + self.g.nx / 2:self.g.ihi + 1,
                     self.g.jlo:self.g.jhi + 1]
        else:
            print(self.g.ilo, self.g.ilo + self.g.nx / 2 + 1)
            L = self[self.g.ilo:self.g.ilo + self.g.nx / 2 + 1,
                     self.g.jlo:self.g.jhi + 1]
            print(self.g.ilo + self.g.nx / 2, self.g.ihi + 2)
            R = self[self.g.ilo + self.g.nx / 2:self.g.ihi + 2,
                     self.g.jlo:self.g.jhi + 1]

        e = abs(L + np.flipud(R)).max()
        print(e, tol, e < tol)
        return e < tol

    def pretty_print(self):
        """
        Print out a small dataset to the screen with the ghost cells
        a different color, to make things stand out
        """

        if self.dtype == np.int:
            fmt = "%4d"
        elif self.dtype == np.float64:
            fmt = "%10.5g"
        else:
            msg.fail("ERROR: dtype not supported")

        # print j descending, so it looks like a grid (y increasing
        # with height)
        for j in reversed(range(self.g.qy)):
            for i in range(self.g.qx):

                if (j < self.g.jlo or j > self.g.jhi or i < self.g.ilo or i > self.g.ihi):
                    gc = 1
                else:
                    gc = 0

                if gc:
                    print("\033[31m" + fmt % (self[i, j]) + "\033[0m", end="")
                else:
                    print(fmt % (self[i, j]), end="")

            print(" ")

        leg = """
         ^ y
         |
         +---> x
        """
        print(leg)


if __name__ == "__main__":

    g = patch.Grid2d(4, 5, ng=1)
    d = np.random.random((g.qx, g.qy))
    a = ArrayIndexer(d, grid=g)

    f = np.random.random((g.qx, g.qy))
    b = ArrayIndexer(f, grid=g)

    a.pretty_print()
    b.pretty_print()
    c = a + b
    c.pretty_print()
    print(c.shape)

    d = np.asarray(c.ip_jp(1, 1))
    print(d.shape)

    print(d)
    c[:, :] = 0.0
    print(d)

    print(type(b))
