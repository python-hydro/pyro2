import numpy

from unsplitFluxes import *
from util import profile
import vars    

def evolve(my_data, dt):

    pf = profile.timer("evolve")
    pf.begin()

    dens = my_data.get_var("density")
    xmom = my_data.get_var("x-momentum")
    ymom = my_data.get_var("y-momentum")
    ener = my_data.get_var("energy")

    grav = my_data.rp.get_param("compressible.grav")

    myg = my_data.grid
        
    Flux_x, Flux_y = unsplitFluxes(my_data, dt)

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
