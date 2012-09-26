from util import runparams
from util import profile
from unsplitFluxes import *
import vars    
import numpy

def evolve(myData, dt):

    pf = profile.timer("evolve")
    pf.begin()

    dens = myData.getVarPtr("density")
    xmom = myData.getVarPtr("x-momentum")
    ymom = myData.getVarPtr("y-momentum")
    ener = myData.getVarPtr("energy")

    grav = runparams.getParam("compressible.grav")

    myg = myData.grid
    
    
    (Flux_x, Flux_y) = unsplitFluxes(myData, dt)
    #Flux_x = numpy.zeros((myg.qx, myg.qy, myData.nvar),  dtype=numpy.float64)
    #Flux_y = numpy.zeros((myg.qx, myg.qy, myData.nvar),  dtype=numpy.float64)

    old_dens = dens.copy()
    old_ymom = ymom.copy()

    # conservative update

    dtdx = dt/myg.dx
    dtdy = dt/myg.dy


    dens[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] += \
        dtdx*(Flux_x[myg.ilo  :myg.ihi+1,myg.jlo  :myg.jhi+1,vars.idens] - \
              Flux_x[myg.ilo+1:myg.ihi+2,myg.jlo  :myg.jhi+1,vars.idens]) + \
        dtdy*(Flux_y[myg.ilo  :myg.ihi+1,myg.jlo  :myg.jhi+1,vars.idens] - \
              Flux_y[myg.ilo  :myg.ihi+1,myg.jlo+1:myg.jhi+2,vars.idens])    

    xmom[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] += \
        dtdx*(Flux_x[myg.ilo  :myg.ihi+1,myg.jlo  :myg.jhi+1,vars.ixmom] - \
              Flux_x[myg.ilo+1:myg.ihi+2,myg.jlo  :myg.jhi+1,vars.ixmom]) + \
        dtdy*(Flux_y[myg.ilo  :myg.ihi+1,myg.jlo  :myg.jhi+1,vars.ixmom] - \
              Flux_y[myg.ilo  :myg.ihi+1,myg.jlo+1:myg.jhi+2,vars.ixmom])    

    ymom[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] += \
        dtdx*(Flux_x[myg.ilo  :myg.ihi+1,myg.jlo  :myg.jhi+1,vars.iymom] - \
              Flux_x[myg.ilo+1:myg.ihi+2,myg.jlo  :myg.jhi+1,vars.iymom]) + \
        dtdy*(Flux_y[myg.ilo  :myg.ihi+1,myg.jlo  :myg.jhi+1,vars.iymom] - \
              Flux_y[myg.ilo  :myg.ihi+1,myg.jlo+1:myg.jhi+2,vars.iymom])    

    ener[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] += \
        dtdx*(Flux_x[myg.ilo  :myg.ihi+1,myg.jlo  :myg.jhi+1,vars.iener] - \
              Flux_x[myg.ilo+1:myg.ihi+2,myg.jlo  :myg.jhi+1,vars.iener]) + \
        dtdy*(Flux_y[myg.ilo  :myg.ihi+1,myg.jlo  :myg.jhi+1,vars.iener] - \
              Flux_y[myg.ilo  :myg.ihi+1,myg.jlo+1:myg.jhi+2,vars.iener])    


    # gravitational source terms
    ymom += 0.5*dt*(dens + old_dens)*grav
    ener += 0.5*dt*(ymom + old_ymom)*grav


    pf.end()
