from util import runparams
from advectiveFluxes import *

def evolve(myData, dt):
    """ evolve the advection equations through one timestep """

    
    dens = myData.getVarPtr("density")

    splitting = runparams.getParam("driver.splitting")

    dtdx = dt/myData.grid.dx
    dtdy = dt/myData.grid.dy

    print "splitting = ", splitting

    if splitting == "unsplit":

        print "here"

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
            
        return 1

    
    # elif splitting == "strang" or splitting == "Strang":

    #     # when we Strang-split, we want to do X(dt) Y(dt) Y(dt) X(dt) a,
    #     # to advance a two timesteps.  We do X Y in even steps, and
    #     # Y X in odd steps to get the right symmetry.
        
    #     if myPatch.nstep % 2 == 0:

    #         flux_x = splitFluxes('x',
    #                              myPatch.nx, myPatch.ny, \
    #                              myPatch.ng, \
    #                              myPatch.dt, myPatch.x, myPatch.y, \
    #                              dens)

    #         myPatch.data[0,0:qx-1,:] = myPatch.data[0,0:qx-1,:] + \
    #                                    dtdx*(flux_x[0:qx-1,:] - flux_x[1:qx,:])


    #         myPatch.fillBC()

    #         flux_y = splitFluxes('y',
    #                              myPatch.nx, myPatch.ny, \
    #                              myPatch.ng, \
    #                              myPatch.dt, myPatch.x, myPatch.y, \
    #                              dens)

    #         myPatch.data[0,:,0:qy-1] = myPatch.data[0,:,0:qy-1] + \
    #                                    dtdy*(flux_y[:,0:qy-1] - flux_y[:,1:qy])

    #     else:
            
    #         flux_y = splitFluxes('y',
    #                              myPatch.nx, myPatch.ny, \
    #                              myPatch.ng, \
    #                              myPatch.dt, myPatch.x, myPatch.y, \
    #                              dens)

    #         myPatch.data[0,:,0:qy-1] = myPatch.data[0,:,0:qy-1] + \
    #                                    dtdy*(flux_y[:,0:qy-1] - flux_y[:,1:qy])
            

    #         myPatch.fillBC()


    #         flux_x = splitFluxes('x',
    #                              myPatch.nx, myPatch.ny, \
    #                              myPatch.ng, \
    #                              myPatch.dt, myPatch.x, myPatch.y, \
    #                              dens)

    #         myPatch.data[0,0:qx-1,:] = myPatch.data[0,0:qx-1,:] + \
    #                                    dtdx*(flux_x[0:qx-1,:] - flux_x[1:qx,:])


    #     return 1

    else:
        print "ERROR: splitting " + splitting + " not supported."
        return -1
    
    
