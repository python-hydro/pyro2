from util import runparams
from advectiveFluxes import *

def evolve(myData, dt):
    """ evolve the advection equations through one timestep """

    
    dens = myData.getVarPtr("density")

    dtdx = dt/myData.grid.dx
    dtdy = dt/myData.grid.dy

    [flux_x, flux_y] =  unsplitFluxes(myData.grid, dt, dens)

    """
    do the differencing for the fluxes now.  Here, we use slices so we
    avoid slow loops in python.  This is equivalent to:

    myPatch.data[i,j] = myPatch.data[i,j] + \
                            dtdx*(flux_x[i,j] - flux_x[i+1,j]) + \
                            dtdy*(flux_y[i,j] - flux_y[i,j+1])
    """

    qx = myData.grid.qx
    qy = myData.grid.qy
    dens[0:qx-1,0:qy-1] = dens[0:qx-1,0:qy-1] + \
        dtdx*(flux_x[0:qx-1,0:qy-1] - flux_x[1:qx,0:qy-1]) + \
        dtdy*(flux_y[0:qx-1,0:qy-1] - flux_y[0:qx-1,1:qy])
            

    
