import numpy
import math
import vars

smallc = 1.e-10
smallrho = 1.e-10
smallp = 1.e-10

def riemann(idir, myGrid, U_l, U_r):
    """
    Solve riemann shock tube problem for a general equation of
    state using the method of Colella, Glaz, and Ferguson.  See
    Almgren et al. 2010 (the CASTRO paper) for details.
    
    The Riemann problem for the Euler's equation produces 4 regions,
    separated by the three characteristics (u - cs, u, u + cs):
    
    
           u - cs    t    u      u + cs
             \       ^   .       /
              \  *L  |   . *R   /
               \     |  .     /
                \    |  .    /
            L    \   | .   /    R
                  \  | .  /
                   \ |. /
                    \|./
           ----------+----------------> x

    We care about the solution on the axis.  The basic idea is to use
    estimates of the wave speeds to figure out which region we are in,
    and then use jump conditions to evaluate the state there.

    Only density jumps across the u characteristic.  All primitive
    variables jump across the other two.  Special attention is needed
    if a rarefaction spans the axis.  

    """

    F = numpy.zeros((myg.qx, myg.qy, myData.nvar),  dtype=numpy.float64)

    j = myg.jlo-1
    while (j <= myg.jhi+1):
    
        i = myg.ilo-1
        while (i <= myg.ihi+1):

            # primitive variable states
            rho_l  = U_l[i,j,vars.idens]
            u_l    = U_l[i,j,vars.ixmom]/rho_l
            v_l    = U_l[i,j,vars.iymom]/rho_l
            rhoe_l = U_l[i,j,vars.iener] - 0.5*rho_l*(u_l**2 + v_l**2)
            e_l = rhoe_l/rho_l
            p_l   = eos.pres(rho_l, e_l)

            rho_r  = U_r[i,j,vars.idens]
            u_r    = U_r[i,j,vars.ixmom]/rho_r
            v_r    = U_r[i,j,vars.iymom]/rho_r
            rhoe_r = U_r[i,j,vars.iener] - 0.5*rho_r*(u_r**2 + v_r**2)
            e_r = rhoe_r/rho_r
            p_r   = eos.pres(rho_r, e_r)

            
            # un = normal velocity
            # ut = transverse velocity
            if (idir == 1):
                un_l = u_l
                ut_l = v_l
            else:
                un_l = v_l
                ut_l = u_l


            # define the Lagrangian sound speed
            W_l = max(smallrho*smallc, math.sqrt(gamma*p_l*rho_l))
            W_r = max(smallrho*smallc, math.sqrt(gamma*p_r*rho_r))

            # and the regular sound speeds
            c_l = max(smallc, math.sqrt(gamma*p_l/rho_l))
            c_r = max(smallc, math.sqrt(gamma*p_r/rho_r))

            # define the star states
            pstar = (W_l*p_r + W_r*p_l + W_l*W_r*(un_l - un_r))/(W_l + W_r)
            pstar = max(pstar, smallp)
            ustar = (W_l*un_l + W_r*un_r + (p_l - p_r))/(W_l + W_r)

            # now compute the remaining state to the left and right
            # of the contact (in the star region)
            rhostar_l = rho_l + (pstar - p_l)/c_l**2
            rhostar_r = rho_r + (pstar - p_r)/c_r**2

            rhoestar_l = rhoe_l + \
                (pstar - p_l)*(rhoe_l/rho_l + p_l/rho_l)/c_l**2
            rhoestar_r = rhoe_r + \
                (pstar - p_r)*(rhoe_r/rho_r + p_r/rho_r)/c_r**2

            cstar_l = max(smallc,math.sqrt(gamma*pstar/rhostar_l))
            cstar_r = max(smallc,math.sqrt(gamma*pstar/rhostar_r))

            # figure out which state we are in, based on the location of
            # the waves
            if (ustar > 0.0):
                
                # contact is moving to the right, we need to understand 
                # the L and *L states

                # Note: transverse velocity only jumps across contact
                ut_state = ut_l

                # define eigenvalues
                lambda_l = un_l - c_l
                lambdastar_l = ustar - cstar_l

                if (pstar > p_l):
                    # the wave is a shock -- find the shock speed
                    sigma = (lambda_l + lambdastar_l)/2.0_dp_t

                    if (sigma > 0.0):
                        # shock is moving to the right -- solution is L state
                        rho_state = rho_l
                        un_state = un_l
                        p_state = p_l
                        rhoe_state = rhoe_l

                    else:
                        # solution is *L state
                        rho_state = rhostar_l
                        un_state = ustar
                        p_state = pstar
                        rhoe_state = rhoestar_l                        

                else:
                    # the wave is a rarefaction
                    if (lambda_l < 0.0 and lambdastar_l < 0.0):
                        # rarefaction fan is moving to the left -- solution is
                        # *L state
                        rho_state = rhostar_l
                        un_state = ustar
                        p_state = pstar
                        rhoe_state = rhoestar_l

                    elif (lambda_l > 0.0 and lambdastar_l > 0.0):
                        # rarefaction fan is moving to the right -- solution is
                        # L state
                        rho_state = rho_l
                        un_state = un_l
                        p_state = p_l
                        rhoe_state = rhoe_l

                    else:
                        # rarefaction spans x/t = 0 -- interpolate
                        alpha = lambda_l/(lambda_l - lambdastar_l)

                        rho_state  = alpha*rhostar_l  + (1.0 - alpha)*rho_l
                        un_state   = alpha*ustar      + (1.0 - alpha)*un_l
                        p_state    = alpha*pstar      + (1.0 - alpha)*p_l
                        rhoe_state = alpha*rhoestar_l + (1.0 - alpha)*rhoe_l

            elif (ustar < 0):
                
                # contact moving left, we need to understand the R and *R
                # states

                # Note: transverse velocity only jumps across contact
                ut_state = ut_r
                
                # define eigenvalues
                lambda_r = un_r + c_r
                lambdastar_r = ustar + cstar_r

                if (pstar > p_r):
                    # the wave if a shock -- find the shock speed 
                    sigma = (lambda_r + lambdastar_r)/2.0_dp_t

                    if (sigma > 0.0):
                        # shock is moving to the right -- solution is *R state
                        rho_state = rhostar_r
                        un_state = ustar
                        p_state = pstar
                        rhoe_state = rhoestar_r

                    else:
                        # solution is R state
                        rho_state = rho_r
                        un_state = un_r
                        p_state = p_r
                        rhoe_state = rhoe_r
                        
                else:
                    # the wave is a rarefaction
                    if (lambda_r < 0.0 and lambdastar_r < 0.0):
                        # rarefaction fan is moving to the left -- solution is
                        # R state
                        rho_state = rho_r
                        un_state = un_r
                        p_state = p_r
                        rhoe_state = rhoe_r
                        
                    elif (lambda_r > 0.0 and lambdastar_r > 0.0):
                        # rarefaction fan is moving to the right -- solution is
                        # *R state
                        rho_state = rhostar_r
                        un_state = ustar
                        p_state = pstar
                        rhoe_state = rhoestar_r
                        
                    else:
                        # rarefaction spans x/t = 0 -- interpolate
                        alpha = lambda_r/(lambda_r - lambdastar_r)

                        rho_state  = alpha*rhostar_r  + (1.0 - alpha)*rho_r
                        un_state   = alpha*ustar      + (1.0 - alpha)*un_r
                        p_state    = alpha*pstar      + (1.0 - alpha)*p_r
                        rhoe_state = alpha*rhoestar_r + (1.0 - alpha)*rhoe_r
                        
            else:  # ustar == 0
                
                rho_state = 0.5*(rhostar_l + rhostar_r)
                un_state = ustar
                ut_state = 0.5*(ut_l + ut_r)
                p_state = pstar
                rhoe_state = 0.5*(rhoestar_l + rhoestar_r)

                        
            # compute the fluxes
            F[i,j,vars.idens] = rho_state*un_state

            if (idir == 1):
                F[i,j,vars.ixmom] = rho_state*un_state**2 + p_state
                F[i,j,vars.iymom] = rho_state*ut_state
            else:
                F[i,j,vars.ixmom] = rho_state*ut_state
                F[i,j,vars.iymom] = rho_state*un_state**2 + p_state

            F[i,j,vars.iener] = rhe_state*u_state + \
                0.5*rho_state*u_state**3 + p_state*u_state

    return F
