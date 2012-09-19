from util import runparams
#from unsplitFluxes import *
from vars import *

def evolve(myData, dt):

    dens = myData.getVarPtr("density")
    xmom = myData.getVarPtr("x-momentum")
    ymom = myData.getVarPtr("y-momentum")
    ener = myData.getVarPtr("energy")

    grav = runparams.getParam("compressible.grav")
    
    
    #(Flux_x, Flux_y) = unsplitFluxes(myData, dt)


    old_dens = dens.copy()
    old_ymom = ymom.copy()

    # conservative update
    myg = myData.grid

    dens[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] += \
        dtdx*(Flux_x[myg.ilo  :myg.ihi+1,myg.jlo  :myg.jhi+1,idens] - \
              Flux_x[myg.ilo+1:myg.ihi+2,myg.jlo  :myg.jhi+1,idens]) + \
        dtdy*(Flux_y[myg.ilo  :myg.ihi+1,myg.jlo  :myg.jhi+1,idens] - \
              Flux_y[myg.ilo  :myg.ihi+1,myg.jlo+1:myg.jhi+2,idens])    

    xmom[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] += \
        dtdx*(Flux_x[myg.ilo  :myg.ihi+1,myg.jlo  :myg.jhi+1,ixmom] - \
              Flux_x[myg.ilo+1:myg.ihi+2,myg.jlo  :myg.jhi+1,ixmom]) + \
        dtdy*(Flux_y[myg.ilo  :myg.ihi+1,myg.jlo  :myg.jhi+1,ixmom] - \
              Flux_y[myg.ilo  :myg.ihi+1,myg.jlo+1:myg.jhi+2,ixmom])    

    ymom[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] += \
        dtdx*(Flux_x[myg.ilo  :myg.ihi+1,myg.jlo  :myg.jhi+1,iymom] - \
              Flux_x[myg.ilo+1:myg.ihi+2,myg.jlo  :myg.jhi+1,iymom]) + \
        dtdy*(Flux_y[myg.ilo  :myg.ihi+1,myg.jlo  :myg.jhi+1,iymom] - \
              Flux_y[myg.ilo  :myg.ihi+1,myg.jlo+1:myg.jhi+2,iymom])    

    ener[myg.ilo:myg.ihi+1,myg.jlo:myg.jhi+1] += \
        dtdx*(Flux_x[myg.ilo  :myg.ihi+1,myg.jlo  :myg.jhi+1,iener] - \
              Flux_x[myg.ilo+1:myg.ihi+2,myg.jlo  :myg.jhi+1,iener]) + \
        dtdy*(Flux_y[myg.ilo  :myg.ihi+1,myg.jlo  :myg.jhi+1,iener] - \
              Flux_y[myg.ilo  :myg.ihi+1,myg.jlo+1:myg.jhi+2,iener])    


    # gravitational source terms
    ymom += 0.5*dt*(dens + old_dens)*grav
    ener += 0.5*dt*(ymom + old_ymom)*grav

