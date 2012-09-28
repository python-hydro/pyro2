subroutine states(idir, qx, qy, ng, dx, dt, &
                  nvar, &
                  gamma, &
                  r, u, v, p, &
                  ldelta_r, ldelta_u, ldelta_v, ldelta_p, &
                  q_l, q_r)

  implicit none

  integer, intent(in) :: idir
  integer, intent(in) :: qx, qy, ng
  double precision, intent(in) :: dx, dt
  integer, intent(in) :: nvar
  double precision, intent(in) :: gamma

  ! 0-based indexing to match python
  double precision, intent(inout) :: r(0:qx-1, 0:qy-1)
  double precision, intent(inout) :: u(0:qx-1, 0:qy-1)
  double precision, intent(inout) :: v(0:qx-1, 0:qy-1)
  double precision, intent(inout) :: p(0:qx-1, 0:qy-1)

  double precision, intent(inout) :: ldelta_r(0:qx-1, 0:qy-1)
  double precision, intent(inout) :: ldelta_u(0:qx-1, 0:qy-1)
  double precision, intent(inout) :: ldelta_v(0:qx-1, 0:qy-1)
  double precision, intent(inout) :: ldelta_p(0:qx-1, 0:qy-1)

  double precision, intent(  out) :: q_l(0:qx-1, 0:qy-1, 0:nvar-1)
  double precision, intent(  out) :: q_r(0:qx-1, 0:qy-1, 0:nvar-1)

!f2py depend(qx, qy) :: r, u, v, p
!f2py depend(qx, qy) :: ldelta_r, ldelta_u, ldelta_v, ldelta_p
!f2py depend(qx, qy, nvar) :: q_l, q_r
!f2py intent(in) :: r, u, v, p
!f2py intent(in) :: ldelta_r, ldelta_u, ldelta_v, ldelta_p
!f2py intent(out) :: q_l, q_r
 


  ! predict the cell-centered state to the edges in one-dimension
  ! using the reconstructed, limited slopes.
  !
  ! We follow the convection here that V_l[i] is the left state at the
  ! i-1/2 interface and V_l[i+1] is the left state at the i+1/2
  ! interface.
  !
  !
  ! We need the left and right eigenvectors and the eigenvalues for
  ! the system projected along the x-direction
  !                                      
  ! Taking our state vector as Q = (rho, u, v, p)^T, the eigenvalues
  ! are u - c, u, u + c. 
  !
  ! We look at the equations of hydrodynamics in a split fashion --
  ! i.e., we only consider one dimension at a time.
  !
  ! Considering advection in the x-direction, the Jacobian matrix for
  ! the primitive variable formulation of the Euler equations
  ! projected in the x-direction is:
  !
  !        / u   r   0   0 \
  !        | 0   u   0  1/r |
  !    A = | 0   0   u   0  |
  !        \ 0  rc^2 0   u  /
  !            
  ! The right eigenvectors are
  !
  !        /  1  \        / 1 \        / 0 \        /  1  \
  !        |-c/r |        | 0 |        | 0 |        | c/r |
  !   r1 = |  0  |   r2 = | 0 |   r3 = | 1 |   r4 = |  0  |
  !        \ c^2 /        \ 0 /        \ 0 /        \ c^2 /
  !
  ! In particular, we see from r3 that the transverse velocity (v in
  ! this case) is simply advected at a speed u in the x-direction.
  !
  ! The left eigenvectors are
  !
  !    l1 =     ( 0,  -r/(2c),  0, 1/(2c^2) )
  !    l2 =     ( 1,     0,     0,  -1/c^2  )
  !    l3 =     ( 0,     0,     1,     0    )
  !    l4 =     ( 0,   r/(2c),  0, 1/(2c^2) )
  !
  ! The fluxes are going to be defined on the left edge of the
  ! computational zones
  !
  !           |             |             |             |
  !           |             |             |             |
  !          -+------+------+------+------+------+------+--
  !           |     i-1     |      i      |     i+1     | 
  !                        ^ ^           ^
  !                    q_l,i q_r,i  q_l,i+1    
  !
  ! q_r,i and q_l,i+1 are computed using the information in zone i,j.

  integer :: ilo, ihi, jlo, jhi
  integer :: nx, ny
  integer :: i, j, m

  double precision :: dq(0:nvar-1), q(0:nvar-1)
  double precision :: lvec(0:nvar-1,0:nvar-1), rvec(0:nvar-1,0:nvar-1)
  double precision :: eval(0:nvar-1)
  double precision :: betal(0:nvar-1), betar(0:nvar-1)

  double precision :: dtdx, dtdx4
  double precision :: cs

  double precision :: sum, sum_l, sum_r, factor

  nx = qx - 2*ng; ny = qy - 2*ng
  ilo = ng; ihi = ng+nx-1; jlo = ng; jhi = ng+ny-1

  dtdx = dt/dx
  dtdx4 = 0.25d0*dtdx

  ! this is the loop over zones.  For zone i, we see q_l[i+1] and q_r[i]
  do j = jlo-2, jhi+2
     do i = ilo-2, ihi+2

        dq(:) = [ldelta_r(i,j), &
                 ldelta_u(i,j), & 
                 ldelta_v(i,j), &
                 ldelta_p(i,j)]
        
        q(:) = [r(i,j), u(i,j), v(i,j), p(i,j)]

        cs = sqrt(gamma*p(i,j)/r(i,j))

        ! compute the eigenvalues and eigenvectors
        if (idir == 1) then
           eval(:) = [u(i,j) - cs, u(i,j), u(i,j), u(i,j) + cs]
        
           lvec(0,:) = [ 0.0d0, -0.5d0*r(i,j)/cs, 0.0d0, 0.5d0/(cs*cs)  ]
           lvec(1,:) = [ 1.0d0, 0.0d0,            0.0d0, -1.0d0/(cs*cs) ]
           lvec(2,:) = [ 0.0d0, 0.0d0,            1.0d0, 0.0d0          ]
           lvec(3,:) = [ 0.0d0, 0.5d0*r(i,j)/cs,  0.0d0, 0.5d0/(cs*cs)  ]

           rvec(0,:) = [1.0d0, -cs/r(i,j), 0.0d0, cs*cs ]
           rvec(1,:) = [1.0d0, 0.0d0,      0.0d0, 0.0d0 ]
           rvec(2,:) = [0.0d0, 0.0d0,      1.0d0, 0.0d0 ]
           rvec(3,:) = [1.0d0, cs/r(i,j),  0.0d0, cs*cs ]

        else
           eval = [v(i,j) - cs, v(i,j), v(i,j), v(i,j) + cs]
                            
           lvec(0,:) = [ 0.0d0, 0.0d0, -0.5d0*r(i,j)/cs, 0.5d0/(cs*cs)  ]
           lvec(1,:) = [ 1.0d0, 0.0d0, 0.0d0,            -1.0d0/(cs*cs) ]
           lvec(2,:) = [ 0.0d0, 1.0d0, 0.0d0,            0.0d0          ]
           lvec(3,:) = [ 0.0d0, 0.0d0, 0.5d0*r(i,j)/cs,  0.5d0/(cs*cs)  ]

           rvec(0,:) = [1.0d0, 0.0d0, -cs/r(i,j), cs*cs ]
           rvec(1,:) = [1.0d0, 0.0d0, 0.0d0,      0.0d0 ]
           rvec(2,:) = [0.0d0, 1.0d0, 0.0d0,      0.0d0 ]
           rvec(3,:) = [1.0d0, 0.0d0, cs/r(i,j),  cs*cs ]

        endif

        ! define the reference states
        if (idir == 1) then
           ! this is one the right face of the current zone,
           ! so the fastest moving eigenvalue is eval[3] = u + c
           factor = 0.5d0*(1.0d0 - dtdx*max(eval(3), 0.0d0))
           q_l(i+1,j,:) = q(:) + factor*dq(:)
               
           ! left face of the current zone, so the fastest moving
           ! eigenvalue is eval[3] = u - c
           factor = 0.5d0*(1.0d0 + dtdx*min(eval(0), 0.0d0))
           q_r(i,  j,:) = q(:) - factor*dq(:)
    
        else

           factor = 0.5d0*(1.0d0 - dtdx*max(eval(3), 0.0d0))
           q_l(i,j+1,:) = q(:) + factor*dq(:)

           factor = 0.5d0*(1.0d0 + dtdx*min(eval(0), 0.0d0))
           q_r(i,j,  :) = q(:) - factor*dq(:)

        endif

        ! compute the Vhat functions
        do m = 0, 3
           sum = dot_product(lvec(m,:),dq(:))

           betal(m) = dtdx4*(eval(3) - eval(m))*(sign(1.0d0,eval(m)) + 1.0d0)*sum
           betar(m) = dtdx4*(eval(0) - eval(m))*(1.0d0 - sign(1.0d0,eval(m)))*sum
        enddo

        ! construct the states
        do m = 0, 3
           sum_l = dot_product(betal(:),rvec(:,m))
           sum_r = dot_product(betar(:),rvec(:,m))

           if (idir == 1) then
              q_l(i+1,j,m) = q_l(i+1,j,m) + sum_l
              q_r(i,  j,m) = q_r(i,  j,m) + sum_r
           else
              q_l(i,j+1,m) = q_l(i,j+1,m) + sum_l
              q_r(i,j,  m) = q_r(i,j,  m) + sum_r
           endif

        enddo

     enddo
  enddo

end subroutine states


subroutine riemann_cgf(idir, qx, qy, ng, &
                       nvar, idens, ixmom, iymom, iener, &
                       gamma, U_l, U_r, F)

  implicit none

  integer, intent(in) :: idir
  integer, intent(in) :: qx, qy, ng
  integer, intent(in) :: nvar, idens, ixmom, iymom, iener
  double precision, intent(in) :: gamma

  ! 0-based indexing to match python 
  double precision, intent(inout) :: U_l(0:qx-1,0:qy-1,0:nvar-1)
  double precision, intent(inout) :: U_r(0:qx-1,0:qy-1,0:nvar-1)
  double precision, intent(  out) :: F(0:qx-1,0:qy-1,0:nvar-1)

!f2py depend(qx, qy, nvar) :: U_l, U_r     
!f2py intent(in) :: U_l, U_r
!f2py intent(out) :: F

  ! Solve riemann shock tube problem for a general equation of
  ! state using the method of Colella, Glaz, and Ferguson.  See
  ! Almgren et al. 2010 (the CASTRO paper) for details.
  !
  ! The Riemann problem for the Euler's equation produces 4 regions,
  ! separated by the three characteristics (u - cs, u, u + cs):
  !
  !
  !        u - cs    t    u      u + cs
  !          \       ^   .       /
  !           \  *L  |   . *R   /
  !            \     |  .     /
  !             \    |  .    /
  !         L    \   | .   /    R
  !               \  | .  /
  !                \ |. /
  !                 \|./
  !        ----------+----------------> x
  !
  ! We care about the solution on the axis.  The basic idea is to use
  ! estimates of the wave speeds to figure out which region we are in,
  ! and then use jump conditions to evaluate the state there.
  !
  ! Only density jumps across the u characteristic.  All primitive
  ! variables jump across the other two.  Special attention is needed
  ! if a rarefaction spans the axis.  

  integer :: ilo, ihi, jlo, jhi
  integer :: nx, ny
  integer :: i, j

  double precision, parameter :: smallc = 1.e-10
  double precision, parameter :: smallrho = 1.e-10
  double precision, parameter :: smallp = 1.e-10

  double precision :: rho_l, un_l, ut_l, rhoe_l, p_l
  double precision :: rho_r, un_r, ut_r, rhoe_r, p_r

  double precision :: rhostar_l, rhostar_r, rhoestar_l, rhoestar_r
  double precision :: ustar, pstar, cstar_l, cstar_r
  double precision :: lambda_l, lambdastar_l, lambda_r, lambdastar_r
  double precision :: W_l, W_r, c_l, c_r, sigma
  double precision :: alpha

  double precision :: rho_state, un_state, ut_state, p_state, rhoe_state
  

  nx = qx - 2*ng; ny = qy - 2*ng
  ilo = ng; ihi = ng+nx-1; jlo = ng; jhi = ng+ny-1

  do j = jlo-1, jhi+1
     do i = ilo-1, ihi+1

        ! primitive variable states
        rho_l  = U_l(i,j,idens)

        ! un = normal velocity; ut = transverse velocity        
        if (idir == 1) then
           un_l    = U_l(i,j,ixmom)/rho_l
           ut_l    = U_l(i,j,iymom)/rho_l
        else
           un_l    = U_l(i,j,iymom)/rho_l
           ut_l    = U_l(i,j,ixmom)/rho_l
        endif

        rhoe_l = U_l(i,j,iener) - 0.5*rho_l*(un_l**2 + ut_l**2)

        p_l   = rhoe_l*(gamma - 1.0d0)
        p_l = max(p_l, smallp)

        rho_r  = U_r(i,j,idens)

        if (idir == 1) then
           un_r    = U_r(i,j,ixmom)/rho_r
           ut_r    = U_r(i,j,iymom)/rho_r
        else
           un_r    = U_r(i,j,iymom)/rho_r
           ut_r    = U_r(i,j,ixmom)/rho_r
        endif

        rhoe_r = U_r(i,j,iener) - 0.5*rho_r*(un_r**2 + ut_r**2)

        p_r   = rhoe_r*(gamma - 1.0d0)
        p_r = max(p_r, smallp)
            

        ! define the Lagrangian sound speed
        W_l = max(smallrho*smallc, sqrt(gamma*p_l*rho_l))
        W_r = max(smallrho*smallc, sqrt(gamma*p_r*rho_r))

        ! and the regular sound speeds
        c_l = max(smallc, sqrt(gamma*p_l/rho_l))
        c_r = max(smallc, sqrt(gamma*p_r/rho_r))

        ! define the star states
        pstar = (W_l*p_r + W_r*p_l + W_l*W_r*(un_l - un_r))/(W_l + W_r)
        pstar = max(pstar, smallp)
        ustar = (W_l*un_l + W_r*un_r + (p_l - p_r))/(W_l + W_r)

        ! now compute the remaining state to the left and right
        ! of the contact (in the star region)
        rhostar_l = rho_l + (pstar - p_l)/c_l**2
        rhostar_r = rho_r + (pstar - p_r)/c_r**2

        rhoestar_l = rhoe_l + &
             (pstar - p_l)*(rhoe_l/rho_l + p_l/rho_l)/c_l**2
        rhoestar_r = rhoe_r + &
             (pstar - p_r)*(rhoe_r/rho_r + p_r/rho_r)/c_r**2

        cstar_l = max(smallc,sqrt(gamma*pstar/rhostar_l))
        cstar_r = max(smallc,sqrt(gamma*pstar/rhostar_r))

        ! figure out which state we are in, based on the location of
        ! the waves
        if (ustar > 0.0d0) then
                
           ! contact is moving to the right, we need to understand 
           ! the L and *L states

           ! Note: transverse velocity only jumps across contact
           ut_state = ut_l

           ! define eigenvalues
           lambda_l = un_l - c_l
           lambdastar_l = ustar - cstar_l

           if (pstar > p_l) then
              ! the wave is a shock -- find the shock speed
              sigma = (lambda_l + lambdastar_l)/2.0d0

              if (sigma > 0.0d0) then
                 ! shock is moving to the right -- solution is L state
                 rho_state = rho_l
                 un_state = un_l
                 p_state = p_l
                 rhoe_state = rhoe_l
                 
              else
                 ! solution is *L state
                 rho_state = rhostar_l
                 un_state = ustar
                 p_state = pstar
                 rhoe_state = rhoestar_l                        
              endif

           else
              ! the wave is a rarefaction
              if (lambda_l < 0.0d0 .and. lambdastar_l < 0.0d0) then
                 ! rarefaction fan is moving to the left -- solution is
                 ! *L state
                 rho_state = rhostar_l
                 un_state = ustar
                 p_state = pstar
                 rhoe_state = rhoestar_l
                 
              else if (lambda_l > 0.0d0 .and. lambdastar_l > 0.0d0) then
                 ! rarefaction fan is moving to the right -- solution is
                 ! L state
                 rho_state = rho_l
                 un_state = un_l
                 p_state = p_l
                 rhoe_state = rhoe_l

              else
                 ! rarefaction spans x/t = 0 -- interpolate
                 alpha = lambda_l/(lambda_l - lambdastar_l)

                 rho_state  = alpha*rhostar_l  + (1.0d0 - alpha)*rho_l
                 un_state   = alpha*ustar      + (1.0d0 - alpha)*un_l
                 p_state    = alpha*pstar      + (1.0d0 - alpha)*p_l
                 rhoe_state = alpha*rhoestar_l + (1.0d0 - alpha)*rhoe_l
              endif

           endif

        else if (ustar < 0) then
                
           ! contact moving left, we need to understand the R and *R
           ! states

           ! Note: transverse velocity only jumps across contact
           ut_state = ut_r
                
           ! define eigenvalues
           lambda_r = un_r + c_r
           lambdastar_r = ustar + cstar_r

           if (pstar > p_r) then
              ! the wave if a shock -- find the shock speed 
              sigma = (lambda_r + lambdastar_r)/2.0d0

              if (sigma > 0.0d0) then
                 ! shock is moving to the right -- solution is *R state
                 rho_state = rhostar_r
                 un_state = ustar
                 p_state = pstar
                 rhoe_state = rhoestar_r

              else
                 ! solution is R state
                 rho_state = rho_r
                 un_state = un_r
                 p_state = p_r
                 rhoe_state = rhoe_r
              endif

           else
              ! the wave is a rarefaction
              if (lambda_r < 0.0d0 .and. lambdastar_r < 0.0d0) then
                 ! rarefaction fan is moving to the left -- solution is
                 ! R state
                 rho_state = rho_r
                 un_state = un_r
                 p_state = p_r
                 rhoe_state = rhoe_r
                        
              else if (lambda_r > 0.0d0 .and. lambdastar_r > 0.0d0) then
                 ! rarefaction fan is moving to the right -- solution is
                 ! *R state
                 rho_state = rhostar_r
                 un_state = ustar
                 p_state = pstar
                 rhoe_state = rhoestar_r
                 
              else
                 ! rarefaction spans x/t = 0 -- interpolate
                 alpha = lambda_r/(lambda_r - lambdastar_r)

                 rho_state  = alpha*rhostar_r  + (1.0d0 - alpha)*rho_r
                 un_state   = alpha*ustar      + (1.0d0 - alpha)*un_r
                 p_state    = alpha*pstar      + (1.0d0 - alpha)*p_r
                 rhoe_state = alpha*rhoestar_r + (1.0d0 - alpha)*rhoe_r
                        
              endif
              
           endif
           
        else  ! ustar == 0
                
           rho_state = 0.5*(rhostar_l + rhostar_r)
           un_state = ustar
           ut_state = 0.5*(ut_l + ut_r)
           p_state = pstar
           rhoe_state = 0.5*(rhoestar_l + rhoestar_r)

        endif

        ! compute the fluxes
        F(i,j,idens) = rho_state*un_state

        if (idir == 1) then
           F(i,j,ixmom) = rho_state*un_state**2 + p_state
           F(i,j,iymom) = rho_state*ut_state*un_state
        else
           F(i,j,ixmom) = rho_state*ut_state*un_state
           F(i,j,iymom) = rho_state*un_state**2 + p_state
        endif

        F(i,j,iener) = rhoe_state*un_state + &
             0.5*rho_state*(un_state**2 + ut_state**2)*un_state + &
             p_state*un_state

     enddo
  enddo

end subroutine riemann_cgf


subroutine riemann_HLLC(idir, qx, qy, ng, &
                        nvar, idens, ixmom, iymom, iener, &
                        gamma, U_l, U_r, F)


  implicit none

  integer, intent(in) :: idir
  integer, intent(in) :: qx, qy, ng
  integer, intent(in) :: nvar, idens, ixmom, iymom, iener
  double precision, intent(in) :: gamma

  ! 0-based indexing to match python 
  double precision, intent(inout) :: U_l(0:qx-1,0:qy-1,0:nvar-1)
  double precision, intent(inout) :: U_r(0:qx-1,0:qy-1,0:nvar-1)
  double precision, intent(  out) :: F(0:qx-1,0:qy-1,0:nvar-1)

!f2py depend(qx, qy, nvar) :: U_l, U_r     
!f2py intent(in) :: U_l, U_r
!f2py intent(out) :: F

  ! this is the HLLC Riemann solver.  The implementation follows
  ! directly out of Toro's book.  Note: this does not handle the
  ! transonic rarefaction.

  integer :: ilo, ihi, jlo, jhi
  integer :: nx, ny
  integer :: i, j

  double precision, parameter :: smallc = 1.e-10
  double precision, parameter :: smallrho = 1.e-10
  double precision, parameter :: smallp = 1.e-10

  double precision :: rho_l, un_l, ut_l, rhoe_l, p_l
  double precision :: rho_r, un_r, ut_r, rhoe_r, p_r

  double precision :: rhostar_l, rhostar_r, rho_avg
  double precision :: ustar, pstar
  double precision :: Q, p_min, p_max, p_lr, p_guess
  double precision :: factor, factor2
  double precision :: g_l, g_r, A_l, B_l, A_r, B_r, z
  double precision :: S_l, S_r, S_c
  double precision :: c_l, c_r, c_avg
  
  double precision :: U_state(0:nvar-1)
  double precision :: HLLCfactor

  

  nx = qx - 2*ng; ny = qy - 2*ng
  ilo = ng; ihi = ng+nx-1; jlo = ng; jhi = ng+ny-1

  do j = jlo-1, jhi+1
     do i = ilo-1, ihi+1

        ! primitive variable states
        rho_l  = U_l(i,j,idens)

        ! un = normal velocity; ut = transverse velocity        
        if (idir == 1) then
           un_l    = U_l(i,j,ixmom)/rho_l
           ut_l    = U_l(i,j,iymom)/rho_l
        else
           un_l    = U_l(i,j,iymom)/rho_l
           ut_l    = U_l(i,j,ixmom)/rho_l
        endif

        rhoe_l = U_l(i,j,iener) - 0.5*rho_l*(un_l**2 + ut_l**2)

        p_l   = rhoe_l*(gamma - 1.0d0)
        p_l = max(p_l, smallp)

        rho_r  = U_r(i,j,idens)

        if (idir == 1) then
           un_r    = U_r(i,j,ixmom)/rho_r
           ut_r    = U_r(i,j,iymom)/rho_r
        else
           un_r    = U_r(i,j,iymom)/rho_r
           ut_r    = U_r(i,j,ixmom)/rho_r
        endif

        rhoe_r = U_r(i,j,iener) - 0.5*rho_r*(un_r**2 + ut_r**2)

        p_r   = rhoe_r*(gamma - 1.0d0)
        p_r = max(p_r, smallp)
            
        
        ! compute the sound speeds
        c_l = max(smallc, sqrt(gamma*p_l/rho_l))
        c_r = max(smallc, sqrt(gamma*p_r/rho_r))

        ! Estimate the star quantities -- use one of three methods to
        ! do this -- the primitive variable Riemann solver, the two
        ! shock approximation, or the two rarefaction approximation.
        ! Pick the method based on the pressure states at the
        ! interface.
        
        p_max = max(p_l, p_r)
        p_min = min(p_l, p_r)

        Q = p_max/p_min

        rho_avg = 0.5*(rho_l + rho_r)
        c_avg = 0.5*(c_l + c_r)

        ! primitive variable Riemann solver (Toro, 9.3)
        factor = rho_avg*c_avg
        factor2 = rho_avg/c_avg

        pstar = 0.5*(p_l + p_r) + 0.5*(un_l - un_r)*factor
        ustar = 0.5*(un_l + un_r) + 0.5*(p_l - p_r)/factor

        rhostar_l = rho_l + (un_l - ustar)*factor2
        rhostar_r = rho_r + (ustar - un_r)*factor2

        if (Q > 2 .and. (pstar < p_min .or. pstar > p_max)) then

           ! use a more accurate Riemann solver for the estimate here
           
           if (pstar < p_min) then
              
              ! 2-rarefaction Riemann solver
              z = (gamma - 1.0d0)/(2.0d0*gamma)
              p_lr = (p_l/p_r)**z

              ustar = (p_lr*un_l/c_l + un_r/c_r + &
                        2.0d0*(p_lr - 1.0d0)/(gamma - 1.0d0)) / &
                      (p_lr/c_l + 1.0d0/c_r)

              pstar = 0.5d0*(p_l*(1.0d0 + (gamma - 1.0d0)*(un_l - ustar)/ &
                                   (2.0d0*c_l) )**(1.0d0/z) + &
                             p_r*(1.0d0 + (gamma - 1.0d0)*(ustar - un_r)/ &
                                   (2.0d0*c_r) )**(1.0d0/z) )

              rhostar_l = rho_l*(pstar/p_l)**(1.0d0/gamma)
              rhostar_r = rho_r*(pstar/p_r)**(1.0d0/gamma)

           else

              ! 2-shock Riemann solver
              A_r = 2.0/((gamma + 1.0d0)*rho_r)
              B_r = p_r*(gamma - 1.0d0)/(gamma + 1.0d0)

              A_l = 2.0/((gamma + 1.0d0)*rho_l)
              B_l = p_l*(gamma - 1.0d0)/(gamma + 1.0d0)

              ! guess of the pressure
              p_guess = max(0.0d0, pstar)

              g_l = sqrt(A_l / (p_guess + B_l))
              g_r = sqrt(A_r / (p_guess + B_r))

              pstar = (g_l*p_l + g_r*p_r - (un_r - un_l))/(g_l + g_r)

              ustar = 0.5*(un_l + un_r) + &
                   0.5*( (pstar - p_r)*g_r - (pstar - p_l)*g_l)

              rhostar_l = rho_l*(pstar/p_l + (gamma-1.0d0)/(gamma+1.0d0))/ &
                   ( (gamma-1.0d0)/(gamma+1.0d0)*(pstar/p_l) + 1.0d0)

              rhostar_r = rho_r*(pstar/p_r + (gamma-1.0d0)/(gamma+1.0d0))/ &
                   ( (gamma-1.0d0)/(gamma+1.0d0)*(pstar/p_r) + 1.0d0)

           endif
        endif

        ! estimate the nonlinear wave speeds

        if (pstar <= p_l) then
           ! rarefaction
           S_l = un_l - c_l
        else
           ! shock
           S_l = un_l - c_l*sqrt(1.0d0 + ((gamma+1.0d0)/(2.0d0*gamma))* &
                                   (pstar/p_l - 1.0d0))
        endif

        if (pstar <= p_r) then
           ! rarefaction
           S_r = un_r + c_r
        else
           ! shock
           S_r = un_r + c_r*sqrt(1.0d0 + ((gamma+1.0d0)/(2.0d0/gamma))* &
                                  (pstar/p_r - 1.0d0))
        endif

        !  We could just take S_c = u_star as the estimate for the
        !  contact speed, but we can actually do this more accurately
        !  by using the Rankine-Hugonoit jump conditions across each
        !  of the waves (see Toro 10.58, Batten et al. SIAM
        !  J. Sci. and Stat. Comp., 18:1553 (1997)
        S_c = (p_r - p_l + rho_l*un_l*(S_l - un_l) - rho_r*un_r*(S_r - un_r))/ &
             (rho_l*(S_l - un_l) - rho_r*(S_r - un_r))

        
        ! figure out which region we are in and compute the state and 
        ! the interface fluxes using the HLLC Riemann solver
        if (S_r <= 0.0d0) then
           ! R region
           U_state(:) = U_r(i,j,:)

           call consFlux(idir, gamma, idens, ixmom, iymom, iener, nvar, &
                         U_state, F(i,j,:))

        else if (S_r > 0.0d0 .and. S_c <= 0) then
           ! R* region
           HLLCfactor = rho_r*(S_r - un_r)/(S_r - S_c)
           
           U_state(idens) = HLLCfactor

           if (idir == 1) then
              U_state(ixmom) = HLLCfactor*S_c
              U_state(iymom) = HLLCfactor*ut_r
           else
              U_state(ixmom) = HLLCfactor*ut_r
              U_state(iymom) = HLLCfactor*S_c
           endif

           U_state(iener) = HLLCfactor*(U_r(i,j,iener)/rho_r + &
                (S_c - un_r)*(S_c + p_r/(rho_r*(S_r - un_r))))

           ! find the flux on the right interface
           call consFlux(idir, gamma, idens, ixmom, iymom, iener, nvar, &
                         U_r(i,j,:), F(i,j,:))

           ! correct the flux
           F(i,j,:) = F(i,j,:) + S_r*(U_state(:) - U_r(i,j,:))

        else if (S_c > 0.0d0 .and. S_l < 0.0) then
           ! L* region
           HLLCfactor = rho_l*(S_l - un_l)/(S_l - S_c)

           U_state(idens) = HLLCfactor

           if (idir == 1) then
              U_state(ixmom) = HLLCfactor*S_c
              U_state(iymom) = HLLCfactor*ut_l
           else
              U_state(ixmom) = HLLCfactor*ut_l
              U_state(iymom) = HLLCfactor*S_c
           endif

           U_state(iener) = HLLCfactor*(U_l(i,j,iener)/rho_l + &
                (S_c - un_l)*(S_c + p_l/(rho_l*(S_l - un_l))))

           ! find the flux on the left interface
           call consFlux(idir, gamma, idens, ixmom, iymom, iener, nvar, &
                         U_l(i,j,:), F(i,j,:))

           ! correct the flux
           F(i,j,:) = F(i,j,:) + S_l*(U_state(:) - U_l(i,j,:))

        else
           ! L region
           U_state(:) = U_l(i,j,:)

           call consFlux(idir, gamma, idens, ixmom, iymom, iener, nvar, &
                         U_state, F(i,j,:))

        endif
           
     enddo
  enddo
end subroutine riemann_HLLC
   
subroutine consFlux(idir, gamma, idens, ixmom, iymom, iener, nvar, U_state, F)        
  
  integer, intent(in) :: idir
  double precision, intent(in) :: gamma
  integer, intent(in) :: idens, ixmom, iymom, iener, nvar
  double precision, intent(in) :: U_state(0:nvar-1)
  double precision, intent(out) :: F(0:nvar-1)

  double precision :: p, u, v

  u = U_state(ixmom)/U_state(idens)
  v = U_state(iymom)/U_state(idens)

  p = (U_state(iener) - 0.5d0*U_state(idens)*(u*u + v*v))*(gamma - 1.0d0)

  if (idir == 1) then
     F(idens) = U_state(idens)*u
     F(ixmom) = U_state(ixmom)*u + p
     F(iymom) = U_state(iymom)*u
     F(iener) = (U_state(iener) + p)*u
  else
     F(idens) = U_state(idens)*v
     F(ixmom) = U_state(ixmom)*v 
     F(iymom) = U_state(iymom)*v + p
     F(iener) = (U_state(iener) + p)*v
  endif

end subroutine consFlux
  

subroutine artificial_viscosity(qx, qy, ng, dx, dy, &
                                cvisc, u, v, avisco_x, avisco_y)

  implicit none
  integer, intent(in) :: qx, qy, ng
  double precision, intent(in) :: dx, dy
  double precision, intent(in) :: cvisc

  ! 0-based indexing to match python
  double precision, intent(in) :: u(0:qx-1, 0:qy-1)
  double precision, intent(in) :: v(0:qx-1, 0:qy-1)
  double precision, intent(out) :: avisco_x(0:qx-1, 0:qy-1)
  double precision, intent(out) :: avisco_y(0:qx-1, 0:qy-1)

!f2py depend(qx, qy) :: u, v
!f2py depend(qx, qy) :: avisco_x, avisco_y
!f2py intent(in) :: u, v
!f2py intent(out) :: avisco_x, avisco_y

  ! compute the artifical viscosity.  Here, we compute edge-centered
  ! approximations to the divergence of the velocity.  This follows 
  ! directly Colella & Woodward (1984) Eq. 4.5
  !
  ! data locations:
  !
  !   j+3/2--+---------+---------+---------+
  !          |         |         |         |
  !     j+1  +         |         |         |
  !          |         |         |         |
  !   j+1/2--+---------+---------+---------+
  !          |         |         |         |
  !        j +         X         |         |
  !          |         |         |         |
  !   j-1/2--+---------+----Y----+---------+ 
  !          |         |         |         |
  !      j-1 +         |         |         |
  !          |         |         |         | 
  !   j-3/2--+---------+---------+---------+
  !          |    |    |    |    |    |    | 
  !              i-1        i        i+1   
  !        i-3/2     i-1/2     i+1/2     i+3/2 
  !
  ! X is the location of avisco_x(i,j)
  ! Y is the location of avisco_y(i,j)

  integer :: ilo, ihi, jlo, jhi
  integer :: nx, ny
 
  integer :: i, j

  double precision :: divU_x, divU_y

  nx = qx - 2*ng; ny = qy - 2*ng
  ilo = ng; ihi = ng+nx-1; jlo = ng; jhi = ng+ny-1

  do j = jlo-1, jhi+1
     do i = ilo-1, ihi+1

        ! start by computing the divergence on the x-interface.  The
        ! x-difference is simply the difference of the cell-centered
        ! x-velocities on either side of the x-interface.  For the
        ! y-difference, first average the four cells to the node on
        ! each end of the edge, and then difference these to find the
        ! edge centered y difference.
        divU_x = (u(i,j) - u(i-1,j))/dx + &
             0.25d0*(v(i,j+1) + v(i-1,j+1) - v(i,j-1) - v(i-1,j-1))/dy

        avisco_x(i,j) = cvisc*max(-divU_x*dx, 0.0d0)

        ! now the y-interface value
        divU_y = 0.25d0*(u(i+1,j) + u(i+1,j-1) - u(i-1,j) - u(i-1,j-1))/dx + &
             (v(i,j) - v(i,j-1))/dy

        avisco_y(i,j) = cvisc*max(-divU_y*dy, 0.0d0)
        
     enddo
  enddo

end subroutine artificial_viscosity
