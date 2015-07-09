# unit tests

import edge_coeffs
import mesh.patch as patch
import numpy as np
from numpy.testing import assert_array_equal

def test_edge_coeffs():
    # make dx = dy = 1 so the normalization is trivial
    g = patch.Grid2d(4, 6, ng=2, xmax=4, ymax=6)
    
    # we'll fill ghost cells ourself
    b = np.arange(g.qx, dtype=np.float64)

    eta1 = g.scratch_array()
    eta1.d[:,:] = b[:,np.newaxis]

    e1 = edge_coeffs.EdgeCoeffs(g, eta1)

    assert_array_equal(e1.x.d[:,g.jc], 
                       np.array([ 0., 0., 1.5, 2.5, 3.5, 4.5, 5.5, 0.]))

    b = np.arange(g.qy, dtype=np.float64)

    eta2 = g.scratch_array()
    eta2.d[:,:] = b[np.newaxis,:]
    
    e2 = edge_coeffs.EdgeCoeffs(g, eta2)

    assert_array_equal(e2.y.d[g.ic,:], 
                       np.array([ 0., 0., 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 0.]))

    
