from __future__ import print_function

import numpy as np

def _buf_split(b):
    try: bxlo, bxhi, bylo, byhi = b
    except:
        try: blo, bhi = b
        except:
            blo = b
            bhi = b
        bxlo = bylo = blo
        bxhi = byhi = bhi
    return bxlo, bxhi, bylo, byhi


class ArrayIndexer(object):
    """ a class that wraps the data region of a single array (d)
        and allows us to easily do array operations like d[i+1,j]
        using the ip() method. """


    # ?? Can we accomplish this a lot easier by subclassing
    # the ndarray?
    # e.g, the InfoArray example here:
    # http://docs.scipy.org/doc/numpy/user/basics.subclassing.html
    def __init__(self, d=None, grid=None):
        self.d = d
        self.g = grid
        s = d.shape
        self.c = len(s)

    def __add__(self, other):
        if isinstance(other, ArrayIndexer):
            return ArrayIndexer(d=self.d + other.d, grid=self.g)
        else:
            return ArrayIndexer(d=self.d + other, grid=self.g)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, ArrayIndexer):
            return ArrayIndexer(d=self.d - other.d, grid=self.g)
        else:
            return ArrayIndexer(d=self.d - other, grid=self.g)

    def __mul__(self, other):
        if isinstance(other, ArrayIndexer):
            return ArrayIndexer(d=self.d * other.d, grid=self.g)
        else:
            return ArrayIndexer(d=self.d * other, grid=self.g)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, ArrayIndexer):
            return ArrayIndexer(d=self.d / other.d, grid=self.g)
        else:
            return ArrayIndexer(d=self.d / other, grid=self.g)

    def __div__(self, other):
        if isinstance(other, ArrayIndexer):
            return ArrayIndexer(d=self.d / other.d, grid=self.g)
        else:
            return ArrayIndexer(d=self.d / other, grid=self.g)

    def __rdiv__(self, other):
        if isinstance(other, ArrayIndexer):
            return ArrayIndexer(d=other.d / self.d, grid=self.g)
        else:
            return ArrayIndexer(d=other / self.d, grid=self.g)

    def __rtruediv__(self, other):
        if isinstance(other, ArrayIndexer):
            return ArrayIndexer(d=other.d / self.d, grid=self.g)
        else:
            return ArrayIndexer(d=other / self.d, grid=self.g)

    def __pow__(self, other):
        return ArrayIndexer(d=self.d**2, grid=self.g)

    def __abs__(self):
        return ArrayIndexer(d=np.abs(self.d), grid=self.g)

    def v(self, buf=0, n=0, s=1):
        return self.ip_jp(0, 0, buf=buf, n=n, s=s)

    def ip(self, shift, buf=0, n=0, s=1):
        return self.ip_jp(shift, 0, buf=buf, n=n, s=s)

    def jp(self, shift, buf=0, n=0, s=1):
        return self.ip_jp(0, shift, buf=buf, n=n, s=s)

    def ip_jp(self, ishift, jshift, buf=0, n=0, s=1):
        bxlo, bxhi, bylo, byhi = _buf_split(buf)

        if self.c == 2:
            return self.d[self.g.ilo-bxlo+ishift:self.g.ihi+1+bxhi+ishift:s,
                          self.g.jlo-bylo+jshift:self.g.jhi+1+byhi+jshift:s]
        else:
            return self.d[self.g.ilo-bxlo+ishift:self.g.ihi+1+bxhi+ishift:s,
                          self.g.jlo-bylo+jshift:self.g.jhi+1+byhi+jshift:s,n]

    def norm(self, n=0):
        """
        find the norm of the quantity (index n) defined on the same grid,
        in the domain's valid region

        """
        if self.c == 2:
            return self.g.norm(self.d)
        else:
            return self.g.norm(self.d[:,:,n])

    def sqrt(self):
        return ArrayIndexer(d=np.sqrt(self.d), grid=self.g)

    def min(self):
        return self.d.min()

    def max(self):
        return self.d.max()

    def copy(self):
        return ArrayIndexer(d=self.d.copy(), grid=self.g)

    def is_symmetric(self, nodal=False, tol=1.e-14):
        if not nodal:
            L = self.d[self.g.ilo:self.g.ilo+self.g.nx/2,
                       self.g.jlo:self.g.jhi+1]
            R = self.d[self.g.ilo+self.g.nx/2:self.g.ihi+1,
                       self.g.jlo:self.g.jhi+1]
        else:
            print(self.g.ilo,self.g.ilo+self.g.nx/2+1)
            L = self.d[self.g.ilo:self.g.ilo+self.g.nx/2+1,
                       self.g.jlo:self.g.jhi+1]
            print(self.g.ilo+self.g.nx/2,self.g.ihi+2)
            R = self.d[self.g.ilo+self.g.nx/2:self.g.ihi+2,
                       self.g.jlo:self.g.jhi+1]


        e = abs(L - np.flipud(R)).max()
        print(e, tol, e < tol)
        return e < tol


    def is_asymmetric(self, nodal=False, tol=1.e-14):
        if not nodal:
            L = self.d[self.g.ilo:self.g.ilo+self.g.nx/2,
                       self.g.jlo:self.g.jhi+1]
            R = self.d[self.g.ilo+self.g.nx/2:self.g.ihi+1,
                       self.g.jlo:self.g.jhi+1]
        else:
            print(self.g.ilo,self.g.ilo+self.g.nx/2+1)
            L = self.d[self.g.ilo:self.g.ilo+self.g.nx/2+1,
                       self.g.jlo:self.g.jhi+1]
            print(self.g.ilo+self.g.nx/2,self.g.ihi+2)
            R = self.d[self.g.ilo+self.g.nx/2:self.g.ihi+2,
                       self.g.jlo:self.g.jhi+1]


        e = abs(L + np.flipud(R)).max()
        print(e, tol, e < tol)
        return e < tol


    def pretty_print(self, fmt=None):
        """
        Print out a small dataset to the screen with the ghost cells
        a different color, to make things stand out
        """

        if fmt is None:
            if self.d.dtype == np.int:
                fmt = "%4d"
            elif self.d.dtype == np.float64:
                fmt = "%10.5g"
            else:
                msg.fail("ERROR: dtype not supported")


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
                    print("\033[31m" + fmt % (self.d[i,j]) + "\033[0m", end="")
                else:
                    print (fmt % (self.d[i,j]), end="")

            print(" ")

        leg = """
         ^ y
         |
         +---> x
        """
        print(leg)



