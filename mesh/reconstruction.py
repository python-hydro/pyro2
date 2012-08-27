import mesh.patch
import sys
import numpy

def limit(idir, grid, a):

    # second-order MC limiting.  We return ldelta_a, the limited difference
    # of a 

    # a is just a numpy array for a single variable -- for example
    # from the getVarPtr method


    if (grid.ng < 4):
        sys.exit("ERROR: need 4 ghost cells in limit")

    ldelta_a = numpy.zeros((grid.qx, grid.qy), dtype=numpy.float64)

    if (idir == 1):

        j = grid.jlo
        while (j <= grid.jhi):
        
            i = grid.ilo
            while (i <= grid.ihi):

                test = (a[i+1,j] - a[i,j])*(a[i,j] - a[i-1,j])

                if (test > 0.0):
                    ldelta_a[i,j] = min(0.5*numpy.fabs(a[i+1,j] - a[i-1,j]),
                                        min(2.0*numpy.fabs(a[i+1,j] - a[i,j]),
                                            2.0*numpy.fabs(a[i,j] - a[i-1,j]))) * \
                                    numpy.sign(a[i+1,j] - a[i-1,j])

                else:
                     ldelta_a[i,j] = 0.0                       

                i += 1

            j += 1


    elif (idir == 2):

        j = grid.jlo-2
        while (j <= grid.jhi+2):

            i = grid.ilo-2
            while (i <= grid.ihi+2):

                test = (a[i,j+1] - a[i,j])*(a[i,j] - a[i,j-1])

                if (test > 0.0):
                    ldelta_a[i,j] = min(0.5*numpy.fabs(a[i,j+1] - a[i,j-1]),
                                        min(2.0*numpy.fabs(a[i,j+1] - a[i,j]),
                                            2.0*numpy.fabs(a[i,j] - a[i,j-1]))) * \
                                    numpy.sign(a[i,j+1] - a[i,j-1])

                else:
                    ldelta_a[i,j] = 0.0                       

                i += 1

            j += 1

            
    return ldelta_a

    

          
