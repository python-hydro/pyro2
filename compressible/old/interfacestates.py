def interfaceStates(idir, myg, dt, 
                    r, u, v, p, 
                    ldelta_r, ldelta_u, ldelta_v, ldelta_p):
    """
    predict the cell-centered state to the edges in one-dimension using
    the reconstructed, limited slopes.

    We follow the convection here that V_l[i] is the left state at the
    i-1/2 interface and V_l[i+1] is the left state at the i+1/2
    interface.
    """

    V_l = numpy.zeros((myg.qx, myg.qy, vars.nvar),  dtype=numpy.float64)
    V_r = numpy.zeros((myg.qx, myg.qy, vars.nvar),  dtype=numpy.float64)
    

    """
    We need the left and right eigenvectors and the eigenvalues for
    the system projected along the x-direction
                                        
    Taking our state vector as Q = (rho, u, v, p)^T, the eigenvalues
    are u - c, u, u + c. 

    We look at the equations of hydrodynamics in a split fashion --
    i.e., we only consider one dimension at a time.

    Considering advection in the x-direction, the Jacobian matrix for
    the primitive variable formulation of the Euler equations
    projected in the x-direction is:

           / u   r   0   0 \
           | 0   u   0  1/r |
       A = | 0   0   u   0  |
           \ 0  rc^2 0   u  /
              
    The right eigenvectors are

           /  1  \        / 1 \        / 0 \        /  1  \
           |-c/r |        | 0 |        | 0 |        | c/r |
      r1 = |  0  |   r2 = | 0 |   r3 = | 1 |   r4 = |  0  |
           \ c^2 /        \ 0 /        \ 0 /        \ c^2 /

    In particular, we see from r3 that the transverse velocity (v in
    this case) is simply advected at a speed u in the x-direction.

    The left eigenvectors are

       l1 =     ( 0,  -r/(2c),  0, 1/(2c^2) )
       l2 =     ( 1,     0,     0,  -1/c^2  )
       l3 =     ( 0,     0,     1,     0    )
       l4 =     ( 0,   r/(2c),  0, 1/(2c^2) )

    The fluxes are going to be defined on the left edge of the
    computational zones

              |             |             |             |
              |             |             |             |
             -+------+------+------+------+------+------+--
              |     i-1     |      i      |     i+1     | 
                      V_l,i  V_r,i   V_l,i+1    

    V_r,i and V_l,i+1 are computed using the information in zone i,j.

    """

    gamma = runparams.getParam("eos.gamma")

    if (idir == 1):
        dtdx = dt/myg.dx
    else:
        dtdx = dt/myg.dy

    dtdx4 = 0.25*dtdx

    # this is the loop over zones.  For zone i, we see V_l[i+1] and V_r[i]
    j = myg.jlo-2
    while (j <= myg.jhi+2):

        i = myg.ilo-2
        while (i <= myg.ihi+2):

            pfa = profile.timer("interface eval/vec")
            pfa.begin()

            dV = numpy.array([ldelta_r[i,j], ldelta_u[i,j], 
                              ldelta_v[i,j], ldelta_p[i,j]])
            V  = numpy.array([r[i,j], u[i,j], v[i,j], p[i,j]])

            cs = math.sqrt(gamma*p[i,j]/r[i,j])

            # compute the eigenvalues and eigenvectors
            lvec = numpy.zeros((4,4), dtype=numpy.float64)
            rvec = numpy.zeros((4,4), dtype=numpy.float64)

            if (idir == 1):
                eval = numpy.array([u[i,j] - cs, u[i,j], u[i,j], u[i,j] + cs])
                            
                lvec[0,:] = [ 0.0, -0.5*r[i,j]/cs, 0.0, 0.5/(cs*cs)  ]
                lvec[1,:] = [ 1.0, 0.0,            0.0, -1.0/(cs*cs) ]
                lvec[2,:] = [ 0.0, 0.0,            1.0, 0.0          ]
                lvec[3,:] = [ 0.0, 0.5*r[i,j]/cs,  0.0, 0.5/(cs*cs)  ]

                rvec[0,:] = [1.0, -cs/r[i,j], 0.0, cs*cs ]
                rvec[1,:] = [1.0, 0.0,        0.0, 0.0   ]
                rvec[2,:] = [0.0, 0.0,        1.0, 0.0   ]
                rvec[3,:] = [1.0, cs/r[i,j],  0.0, cs*cs ]

            else:
                eval = numpy.array([v[i,j] - cs, v[i,j], v[i,j], v[i,j] + cs])
                            
                lvec[0,:] = [ 0.0, 0.0, -0.5*r[i,j]/cs, 0.5/(cs*cs)  ]
                lvec[1,:] = [ 1.0, 0.0, 0.0,            -1.0/(cs*cs) ]
                lvec[2,:] = [ 0.0, 1.0, 0.0,            0.0          ]
                lvec[3,:] = [ 0.0, 0.0, 0.5*r[i,j]/cs,  0.5/(cs*cs)  ]

                rvec[0,:] = [1.0, 0.0, -cs/r[i,j], cs*cs ]
                rvec[1,:] = [1.0, 0.0, 0.0,        0.0   ]
                rvec[2,:] = [0.0, 1.0, 0.0,        0.0   ]
                rvec[3,:] = [1.0, 0.0, cs/r[i,j],  cs*cs ]



            # define the reference states
            if (idir == 1):
                # this is one the right face of the current zone,
                # so the fastest moving eigenvalue is eval[3] = u + c
                factor = 0.5*(1.0 - dtdx*max(eval[3], 0.0))
                V_l[i+1,j,:] = V[:] + factor*dV[:]
               
                # left face of the current zone, so the fastest moving
                # eigenvalue is eval[3] = u - c
                factor = 0.5*(1.0 + dtdx*min(eval[0], 0.0))
                V_r[i,  j,:] = V[:] - factor*dV[:]
    
            else:

                factor = 0.5*(1.0 - dtdx*max(eval[3], 0.0))
                V_l[i,j+1,:] = V[:] + factor*dV[:]

                factor = 0.5*(1.0 + dtdx*min(eval[0], 0.0))
                V_r[i,j,  :] = V[:] - factor*dV[:]

            pfa.end()

            pfb = profile.timer("states")
            pfb.begin()

            # compute the Vhat functions
            betal = numpy.zeros((4), dtype=numpy.float64)
            betar = numpy.zeros((4), dtype=numpy.float64)

            m = 0
            while (m < 4):
                sum = numpy.dot(lvec[m,:],dV[:])

                betal[m] = dtdx4*(eval[3] - eval[m])*(numpy.sign(eval[m]) + 1.0)*sum
                betar[m] = dtdx4*(eval[0] - eval[m])*(1.0 - numpy.sign(eval[m]))*sum
                m += 1

            # construct the states
            m = 0
            while (m < 4):
                sum_l = numpy.dot(betal[:],rvec[:,m])
                sum_r = numpy.dot(betar[:],rvec[:,m])

                if (idir == 1):
                    V_l[i+1,j,m] += sum_l
                    V_r[i,  j,m] += sum_r
                else:
                    V_l[i,j+1,m] += sum_l
                    V_r[i,j,  m] += sum_r

                m += 1
                
            pfb.end()

            i += 1
        j += 1

    return V_l, V_r

