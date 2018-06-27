subroutine states(idir, qx, qy, ng, dx, dt, &
                  irho, iu, iv, ip, ix, nvar, nspec, &
                  gamma, &
                  qv, dqv, &
                  q_l, q_r)

  implicit none

  integer, intent(in) :: idir
  integer, intent(in) :: qx, qy, ng
  double precision, intent(in) :: dx, dt
  integer, intent(in) :: irho, iu, iv, ip, ix, nvar, nspec
  double precision, intent(in) :: gamma

  ! 0-based indexing to match python
  double precision, intent(inout) :: qv(0:qx-1, 0:qy-1, 0:nvar-1)
  double precision, intent(inout) :: dqv(0:qx-1, 0:qy-1, 0:nvar-1)

  double precision, intent(  out) :: q_l(0:qx-1, 0:qy-1, 0:nvar-1)
  double precision, intent(  out) :: q_r(0:qx-1, 0:qy-1, 0:nvar-1)

!f2py depend(qx, qy, nvar) :: qv, dqv
!f2py depend(qx, qy, nvar) :: q_l, q_r
!f2py intent(in) :: qv, dqv
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
  !        / u   r   0   0 &
  !        | 0   u   0  1/r |
  !    A = | 0   0   u   0  |
  !        & 0  rc^2 0   u  /
  !
  ! The right eigenvectors are
  !
  !        /  1  &        / 1 &        / 0 &        /  1  &
  !        |-c/r |        | 0 |        | 0 |        | c/r |
  !   r1 = |  0  |   r2 = | 0 |   r3 = | 1 |   r4 = |  0  |
  !        & c^2 /        & 0 /        & 0 /        & c^2 /
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
  integer :: i, j, n, m

  double precision :: dq(0:nvar-1), q(0:nvar-1)
  double precision :: lvec(0:nvar-1,0:nvar-1), rvec(0:nvar-1,0:nvar-1)
  double precision :: eval(0:nvar-1)
  double precision :: betal(0:nvar-1), betar(0:nvar-1)

  double precision :: dtdx, dtdx4
  double precision :: cs2, v2, Ap, Am, W, h, Delta

  double precision :: summ, sum_l, sum_r, factor

  integer :: ns

  nx = qx - 2*ng; ny = qy - 2*ng
  ilo = ng; ihi = ng+nx-1; jlo = ng; jhi = ng+ny-1

  ns = nvar - nspec

  dtdx = dt/dx
  dtdx4 = 0.25d0*dtdx

  ! this is the loop over zones.  For zone i, we see q_l[i+1] and q_r[i]
  do j = jlo-2, jhi+2
     do i = ilo-2, ihi+2

        dq(:) = dqv(i,j,:)
        q(:) = qv(i,j,:)

        cs2 = gamma*q(ip)/q(irho)
        v2 = sum(q(iu:iv)**2)
        h = 1 + gamma / (gamma - 1.0d0) * q(ip) / q(irho)

        W = 1 / sqrt(1 - v2)

        lvec(:,:) = 0.0d0
        rvec(:,:) = 0.0d0
        eval(:) = 0.0d0

        ! compute the eigenvalues and eigenvectors
        if (idir == 1) then
           eval(1:2) = [q(iu), q(iu)]
           eval(0) = 1.0d0 / (1.0d0 - v2 * cs2) *&
                (q(iu)*(1.0d0-cs2) - sqrt(cs2) * sqrt((1.0d0-v2) *&
                (1.0d0-v2*cs2 - q(iu)**2*(1.0d0-cs2))))
           eval(3) = 1.0d0 / (1.0d0 - v2 * cs2) * &
                (q(iu)*(1.0d0-cs2) + sqrt(cs2) * sqrt((1.0d0-v2) *&
                (1.0d0-v2*cs2 - q(iu)**2*(1.0d0-cs2))))

           Am = (1.0d0-q(iu)**2) / (1.0d0-q(iu)*eval(0))
           Ap = (1.0d0-q(iu)**2) / (1.0d0-q(iu)*eval(3))

           Delta = h**3*W * (h-1.0d0) * (1.0d0-q(iu)**2) * (Ap*eval(3) - Am*eval(0))

           lvec(0,0:ns-1) = [ h*W*Ap*(q(iu)-eval(3)) - q(iu) - W**2*(v2-q(iu)**2)*&
                (2.0d0*h-1.0d0)*(q(iu)-Ap*eval(3)) + h*Ap*eval(3),&
                1.0d0+W**2*(v2-q(iu)**2)*(2.0d0*h-1.0d0) * (1.0d0-Ap) - h*Ap,&
                W**2*q(iv)*(2.0d0*h-1.0d0)*Ap * (q(iu)-eval(3)),&
                -q(iu) - W**2*(v2-q(iu)**2) *&
                (2.0d0*h-1.0d0)*(q(iu)-Ap*eval(3)) + h * Ap*eval(3)  ]

           lvec(0,0:ns-1) = lvec(0,0:ns-1) * h**2 / Delta

           lvec(1,0:ns-1) = [ h-W, W*q(iu), W*q(iv), -W]
           lvec(1,0:ns-1) = lvec(1,0:ns-1) * W / (h - 1.0d0)

           lvec(2,0:ns-1) = [ -q(iv), q(iu)*q(iv),&
                1.0d0-q(iu), -q(iv) ]

            lvec(2,0:ns-1) = lvec(2,0:ns-1) / (h * (1.0d0-q(iu)**2))

           lvec(3,0:ns-1) = [ h*W*Am*(q(iu)-eval(0)) - q(iu) -&
                W**2 * (v2-q(iu)**2)*(2.0d0*h-1.0d0)*&
                (q(iu)-Am*eval(0)) + h*Am*eval(0),&
                1.0d0+W**2*(v2-q(iu)**2)*(2.0d0*h-1.0d0) *&
                (1.0d0-Am) - h*Am,&
                W**2*q(iv)*(2.0d0*h-1.0d0)*Am * (q(iu)-eval(0)),&
                -q(iu) - W**2*(v2-q(iu)**2) *&
                (2.0d0*h-1.0d0)*(q(iu)-Am*eval(0)) + h * Am*eval(0)]

           lvec(3,0:ns-1) = -lvec(3,0:ns-1) * h**2 / Delta

           rvec(0,0:ns-1) = [1.0d0, h*W*Am*eval(0), h*W*q(iv),&
                h*W*Am-1.0d0]
           rvec(1,0:ns-1) = [1.0d0/W, q(iu), q(iv), 1.0d0-1.0d0/W ]
           rvec(2,0:ns-1) = [W*q(iv), 2.0d0*h*W**2*q(iu)*q(iv),&
                h*(1.0d0 + 2.0d0*W**2*q(iv)**2),&
                2*h*W**2*q(iv)-W*q(iv) ]
           rvec(3,0:ns-1) = [1.0d0, h*W*Ap*eval(3),  h*W*q(iv),&
                h*W*Ap-1.0d0 ]

           ! now the species -- they only have a 1 in their corresponding slot
           eval(ns:) = q(iu)
           do n = ix, ix-1+nspec
              lvec(n,n) = 1.0
              rvec(n,n) = 1.0
           enddo
        else
            eval(1:2) = [q(iv), q(iv)]
            eval(0) = 1.0d0 / (1.0d0 - v2 * cs2) * &
                (q(iv)*(1.0d0-cs2) - sqrt(cs2) * sqrt((1-v2) * &
                (1.0d0-v2*cs2 - q(iv)**2*(1.0d0-cs2))))
            eval(3) = 1.0d0 / (1.0d0 - v2 * cs2) * &
                (q(iv)*(1.0d0-cs2) + sqrt(cs2) * sqrt((1.0d0-v2) *&
                (1.0d0-v2*cs2 - q(iv)**2*(1.0d0-cs2))))

            Am = (1.0d0-q(iv)**2) / (1.0d0-q(iv)*eval(0))
            Ap = (1.0d0-q(iv)**2) / (1.0d0-q(iv)*eval(3))

            Delta = h**3*W * (h-1.0d0) * (1.0d0-q(iv)**2) * (Ap*eval(3) - Am*eval(0))

            lvec(0,0:ns-1) = [ h*W*Ap*(q(iv)-eval(3)) - q(iv) - W**2*(v2-q(iv)**2)*&
                (2.0d0*h-1.0d0)*(q(iv)-Ap*eval(3)) + h*Ap*eval(3),&
                W**2*q(iu)*(2.0d0*h-1.0d0)*Ap * (q(iv)-eval(3)),&
                1.0d0+W**2*(v2-q(iv)**2)*(2.0d0*h-1.0d0) * (1.0d0-Ap) - h*Ap,&
                -q(iv) - W**2*(v2-q(iv)**2) *&
                (2.0d0*h-1.0d0)*(q(iv)-Ap*eval(3)) + h * Ap*eval(3)  ]

            lvec(0,0:ns-1) = lvec(0,0:ns-1) * h**2 / Delta

            lvec(1,0:ns-1) = [ -q(iu), 1.0d0-q(iv), q(iu)*q(iv),&
                -q(iu)]

            lvec(1,0:ns-1) = lvec(1,0:ns-1) / (h*(1-q(iv)**2))

            lvec(2,0:ns-1) = [ h-W, W*q(iu), W*q(iv), -W ]
            lvec(2,0:ns-1) = lvec(2,0:ns-1) * W / (h-1.0d0)

            lvec(3,0:ns-1) = [ h*W*Am*(q(iv)-eval(0)) - q(iv) -&
                W**2 * (v2-q(iv)**2)*(2.0d0*h-1.0d0)*&
                (q(iv)-Am*eval(0)) + h*Am*eval(0),&
                W**2*q(iu)*(2.0d0*h-1.0d0)*Am * (q(iv)-eval(0)),&
                1.0d0+W**2*(v2-q(iv)**2)*(2.0d0*h-1.0d0) * (1.0d0-Am) - h*Am,&
                -q(iv) - W**2*(v2-q(iv)**2) *&
                (2.0d0*h-1.0d0)*(q(iv)-Am*eval(0)) + h * Am*eval(0)    ]

            lvec(3,0:ns-1) = -lvec(3,0:ns-1) * h**2 / Delta

           rvec(0,0:ns-1) = [1.0d0, h*W*q(iu), h*W*Am*eval(0),&
                h*W*Am-1.0d0]
           rvec(1,0:ns-1) = [W*q(iu),&
                h*(1.0d0 + 2.0d0*W**2*q(iu)**2),&
                2.0d0*h*W**2*q(iu)*q(iv), 2*h*W**2*q(iu)-W*q(iu) ]
           rvec(2,0:ns-1) = [1.0d0/W, q(iu), q(iv), 1.0d0-1.0d0/W ]
           rvec(3,0:ns-1) = [1.0d0, h*W*q(iu), h*W*Ap*eval(3),&
                h*W*Ap-1.0d0 ]

           ! now the species -- they only have a 1 in their corresponding slot
           eval(ns:) = q(iv)
           do n = ix, ix-1+nspec
              lvec(n,n) = 1.0
              rvec(n,n) = 1.0
           enddo

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
        do m = 0, nvar-1
           summ = dot_product(lvec(m,:),dq(:))

           betal(m) = dtdx4*(eval(3) - eval(m))*(sign(1.0d0,eval(m)) + 1.0d0)*summ
           betar(m) = dtdx4*(eval(0) - eval(m))*(1.0d0 - sign(1.0d0,eval(m)))*summ
        enddo

        ! construct the states
        do m = 0, nvar-1
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
                        nvar, idens, ixmom, iymom, iener, irhoX, irho, iu, iv, ip, ix, nspec, &
                        lower_solid, upper_solid, &
                        gamma, U_l, U_r, q_l, q_r, F)


  implicit none

  integer, intent(in) :: idir
  integer, intent(in) :: qx, qy, ng
  integer, intent(in) :: nvar, idens, ixmom, iymom, iener, irhoX, irho, iu, iv, ip, ix, nspec
  integer, intent(in) :: lower_solid, upper_solid
  double precision, intent(in) :: gamma

  ! 0-based indexing to match python
  double precision, intent(inout) :: U_l(0:qx-1,0:qy-1,0:nvar-1)
  double precision, intent(inout) :: U_r(0:qx-1,0:qy-1,0:nvar-1)
  double precision, intent(inout) :: q_l(0:qx-1,0:qy-1,0:nvar-1)
  double precision, intent(inout) :: q_r(0:qx-1,0:qy-1,0:nvar-1)
  double precision, intent(  out) :: F(0:qx-1,0:qy-1,0:nvar-1)

  !f2py depend(qx, qy, nvar) :: q_l, q_r, U_l, U_r
  !f2py intent(in) :: q_l, q_r, U_l, U_r
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
  !          &       ^   .       /
  !           &  *L  |   . *R   /
  !            &     |  .     /
  !             &    |  .    /
  !         L    &   | .   /    R
  !               &  | .  /
  !                & |. /
  !                 &|./
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

  double precision :: rho_l, un_l, ut_l, p_l, rhoe_l
  double precision :: rho_r, un_r, ut_r, p_r, rhoe_r
  double precision :: xn(nspec)
  double precision :: rhostar_l, rhostar_r, rhoestar_l, rhoestar_r
  double precision :: ustar, pstar, cstar_l, cstar_r
  double precision :: lambda_l, lambdastar_l, lambda_r, lambdastar_r
  double precision :: W_l, W_r, c_l, c_r, sigma
  double precision :: alpha, v2

  double precision :: rho_state, un_state, ut_state, p_state, rhoe_state
  double precision :: U(0:nvar-1), q(0:nvar-1), W


  nx = qx - 2*ng; ny = qy - 2*ng
  ilo = ng; ihi = ng+nx-1; jlo = ng; jhi = ng+ny-1

  do j = jlo-1, jhi+1
     do i = ilo-1, ihi+1

        ! primitive variable states
        rho_l  = q_l(i,j,irho)

        ! un = normal velocity; ut = transverse velocity
        if (idir == 1) then
           un_l    = q_l(i,j,iu)
           ut_l    = q_l(i,j,iv)
        else
           un_l    = q_l(i,j,iv)
           ut_l    = q_l(i,j,iu)
        endif

        p_l   = q_l(i,j,ip)
        p_l = max(p_l, smallp)
        rhoe_l = p_l / (gamma - 1.0d0)

        rho_r  = q_r(i,j,irho)

        if (idir == 1) then
           un_r    = q_r(i,j,iu)
           ut_r    = q_r(i,j,iv)
        else
           un_r    = q_r(i,j,iv)
           ut_r    = q_r(i,j,iu)
        endif

        p_r   = q_r(i,j,ip)
        p_r = max(p_r, smallp)
        rhoe_r = p_r / (gamma - 1.0d0)

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
           v2 = un_l**2 + ut_l**2
           lambda_l = 1.0d0 / (1.0d0 - v2 * c_l**2) *&
                (un_l*(1.0d0-c_l**2) - c_l * sqrt((1.0d0-v2) *&
                (1.0d0-v2*c_l**2 - un_l**2*(1.0d0-c_l**2))))

           v2 = ustar**2 + ut_l**2
           lambdastar_l = 1.0d0 / (1.0d0 - v2 * cstar_l**2) *&
                (ustar*(1.0d0-cstar_l**2) - &
                cstar_l * sqrt((1.0d0-v2) * &
                (1.0d0-v2*cstar_l**2 - &
                 ustar**2*(1.0d0-cstar_l**2))))

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
           v2 = un_r**2 + ut_r**2
           lambda_r = 1.0d0 / (1.0d0 - v2 * c_r**2) * &
                (un_r*(1.0d0-c_r**2) + c_r * sqrt((1.0d0-v2) *&
                (1.0d0-v2*c_r**2 - un_r**2*(1.0d0-c_r**2))))

           v2 = ustar**2 + ut_r**2
           lambdastar_r = 1.0d0 / (1.0d0 - v2 * cstar_r**2) * &
                (ustar*(1.0d0-cstar_r**2) + &
                cstar_r * sqrt((1.0d0-v2) * &
                (1.0d0-v2*cstar_r**2 - &
                 ustar**2*(1.0d0-cstar_r**2))))

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

        ! species now
        if (nspec > 0) then
           if (ustar > 0.0) then
              xn(:) = q_l(i,j,ix:ix-1+nspec)

           else if (ustar < 0.0) then
              xn(:) = q_r(i,j,ix:ix-1+nspec)
           else
              xn(:) = 0.5d0*(q_l(i,j,ix:ix-1+nspec) + &
                             q_r(i,j,ix:ix-1+nspec))
           endif
        endif

        ! are we on a solid boundary?
        if (idir == 1) then
           if (i == ilo .and. lower_solid == 1) then
              un_state = 0.0
           endif

           if (i == ihi+1 .and. upper_solid == 1) then
              un_state = 0.0
           endif

        else if (idir == 2) then
           if (j == jlo .and. lower_solid == 1) then
              un_state = 0.0
           endif

           if (j == jhi+1 .and. upper_solid == 1) then
              un_state = 0.0
           endif

        endif

        ! Make primitive state

        q(irho) = rho_state
        if (idir == 1) then
            q(iu) = un_state
            q(iv) = ut_state
        else
            q(iu) = ut_state
            q(iv) = un_state
        endif

        q(ip) = p_state

        ! Make conservative state

        W = 1 / sqrt(1 - q(iu)**2 - q(iv)**2)

        U(idens) = rho_state * W
        U(ixmom) = (rho_state + p_state * gamma / (gamma - 1.0d0)) * q(iu) * W**2
        U(iymom) = (rho_state + p_state * gamma / (gamma - 1.0d0)) * q(iv) * W**2
        U(iener) = (rho_state + p_state * gamma / (gamma - 1.0d0)) * W**2 - p_state - U(idens)

        if (nspec > 0) then
            q(ix:ix-1+nspec) = xn(:)
            U(irhoX:irhoX-1+nspec) = xn(:) * U(idens)
        endif

        ! compute the fluxes
        call consFlux(idir, gamma, idens, ixmom, iymom, iener, &
                      irhoX, iu, iv, ip, nvar, nspec, &
                      U, q, F(i, j, :))

     enddo
  enddo

end subroutine riemann_cgf


subroutine riemann_prim(idir, qx, qy, ng, &
                        nvar, irho, iu, iv, ip, iX, nspec, &
                        lower_solid, upper_solid, &
                        gamma, q_l, q_r, q_int)

  ! this is like riemann_cgf, except that it works on a primitive
  ! variable input state and returns the primitive variable interface
  ! state

  implicit none

  integer, intent(in) :: idir
  integer, intent(in) :: qx, qy, ng
  integer, intent(in) :: nvar, irho, iu, iv, ip, iX, nspec
  integer, intent(in) :: lower_solid, upper_solid
  double precision, intent(in) :: gamma

  ! 0-based indexing to match python
  double precision, intent(inout) :: q_l(0:qx-1,0:qy-1,0:nvar-1)
  double precision, intent(inout) :: q_r(0:qx-1,0:qy-1,0:nvar-1)
  double precision, intent(  out) :: q_int(0:qx-1,0:qy-1,0:nvar-1)

!f2py depend(qx, qy, nvar) :: q_l, q_r, q_int
!f2py intent(in) :: q_l, q_r
!f2py intent(out) :: q_int

  ! Solve riemann shock tube problem for a general equation of
  ! state using the method of Colella, Glaz, and Ferguson.  See
  ! Almgren et al. 2010 (the CASTRO paper) for details.
  !
  ! The Riemann problem for the Euler's equation produces 4 regions,
  ! separated by the three characteristics (u - cs, u, u + cs):
  !
  !
  !        u - cs    t    u      u + cs
  !          &       ^   .       /
  !           &  *L  |   . *R   /
  !            &     |  .     /
  !             &    |  .    /
  !         L    &   | .   /    R
  !               &  | .  /
  !                & |. /
  !                 &|./
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

  double precision :: rho_l, un_l, ut_l, p_l
  double precision :: rho_r, un_r, ut_r, p_r
  double precision :: xn(nspec)
  double precision :: rhostar_l, rhostar_r
  double precision :: ustar, pstar, cstar_l, cstar_r
  double precision :: lambda_l, lambdastar_l, lambda_r, lambdastar_r
  double precision :: W_l, W_r, c_l, c_r, sigma, v2
  double precision :: alpha

  double precision :: rho_state, un_state, ut_state, p_state

  nx = qx - 2*ng; ny = qy - 2*ng
  ilo = ng; ihi = ng+nx-1; jlo = ng; jhi = ng+ny-1

  do j = jlo-1, jhi+1
     do i = ilo-1, ihi+1

        ! primitive variable states
        rho_l  = q_l(i,j,irho)

        ! un = normal velocity; ut = transverse velocity
        if (idir == 1) then
           un_l    = q_l(i,j,iu)
           ut_l    = q_l(i,j,iv)
        else
           un_l    = q_l(i,j,iv)
           ut_l    = q_l(i,j,iu)
        endif

        p_l   = q_l(i,j,ip)
        p_l = max(p_l, smallp)

        rho_r  = q_r(i,j,irho)

        if (idir == 1) then
           un_r    = q_r(i,j,iu)
           ut_r    = q_r(i,j,iv)
        else
           un_r    = q_r(i,j,iv)
           ut_r    = q_r(i,j,iu)
        endif

        p_r   = q_r(i,j,ip)
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
           v2 = un_l**2 + ut_l**2
           lambda_l = 1.0d0 / (1.0d0 - v2 * c_l**2) *&
                (un_l*(1.0d0-c_l**2) - c_l * sqrt((1.0d0-v2) *&
                (1.0d0-v2*c_l**2 - un_l**2*(1.0d0-c_l**2))))

           v2 = ustar**2 + ut_l**2
           lambdastar_l = 1.0d0 / (1.0d0 - v2 * cstar_l**2) *&
                (ustar*(1.0d0-cstar_l**2) - &
                cstar_l * sqrt((1.0d0-v2) * &
                (1.0d0-v2*cstar_l**2 - &
                 ustar**2*(1.0d0-cstar_l**2))))

           if (pstar > p_l) then
              ! the wave is a shock -- find the shock speed
              sigma = (lambda_l + lambdastar_l)/2.0d0

              if (sigma > 0.0d0) then
                 ! shock is moving to the right -- solution is L state
                 rho_state = rho_l
                 un_state = un_l
                 p_state = p_l

              else
                 ! solution is *L state
                 rho_state = rhostar_l
                 un_state = ustar
                 p_state = pstar
              endif

           else
              ! the wave is a rarefaction
              if (lambda_l < 0.0d0 .and. lambdastar_l < 0.0d0) then
                 ! rarefaction fan is moving to the left -- solution is
                 ! *L state
                 rho_state = rhostar_l
                 un_state = ustar
                 p_state = pstar

              else if (lambda_l > 0.0d0 .and. lambdastar_l > 0.0d0) then
                 ! rarefaction fan is moving to the right -- solution is
                 ! L state
                 rho_state = rho_l
                 un_state = un_l
                 p_state = p_l

              else
                 ! rarefaction spans x/t = 0 -- interpolate
                 alpha = lambda_l/(lambda_l - lambdastar_l)

                 rho_state  = alpha*rhostar_l  + (1.0d0 - alpha)*rho_l
                 un_state   = alpha*ustar      + (1.0d0 - alpha)*un_l
                 p_state    = alpha*pstar      + (1.0d0 - alpha)*p_l
              endif

           endif

        else if (ustar < 0) then

           ! contact moving left, we need to understand the R and *R
           ! states

           ! Note: transverse velocity only jumps across contact
           ut_state = ut_r

           ! define eigenvalues
           v2 = un_r**2 + ut_r**2
           lambda_r = 1.0d0 / (1.0d0 - v2 * c_r**2) * &
                (un_r*(1.0d0-c_r**2) + c_r * sqrt((1.0d0-v2) *&
                (1.0d0-v2*c_r**2 - un_r**2*(1.0d0-c_r**2))))

           v2 = ustar**2 + ut_r**2
           lambdastar_r = 1.0d0 / (1.0d0 - v2 * cstar_r**2) * &
                (ustar*(1.0d0-cstar_r**2) + &
                cstar_r * sqrt((1.0d0-v2) * &
                (1.0d0-v2*cstar_r**2 - &
                 ustar**2*(1.0d0-cstar_r**2))))

           if (pstar > p_r) then
              ! the wave if a shock -- find the shock speed
              sigma = (lambda_r + lambdastar_r)/2.0d0

              if (sigma > 0.0d0) then
                 ! shock is moving to the right -- solution is *R state
                 rho_state = rhostar_r
                 un_state = ustar
                 p_state = pstar

              else
                 ! solution is R state
                 rho_state = rho_r
                 un_state = un_r
                 p_state = p_r
              endif

           else
              ! the wave is a rarefaction
              if (lambda_r < 0.0d0 .and. lambdastar_r < 0.0d0) then
                 ! rarefaction fan is moving to the left -- solution is
                 ! R state
                 rho_state = rho_r
                 un_state = un_r
                 p_state = p_r

              else if (lambda_r > 0.0d0 .and. lambdastar_r > 0.0d0) then
                 ! rarefaction fan is moving to the right -- solution is
                 ! *R state
                 rho_state = rhostar_r
                 un_state = ustar
                 p_state = pstar

              else
                 ! rarefaction spans x/t = 0 -- interpolate
                 alpha = lambda_r/(lambda_r - lambdastar_r)

                 rho_state  = alpha*rhostar_r  + (1.0d0 - alpha)*rho_r
                 un_state   = alpha*ustar      + (1.0d0 - alpha)*un_r
                 p_state    = alpha*pstar      + (1.0d0 - alpha)*p_r

              endif

           endif

        else  ! ustar == 0

           rho_state = 0.5*(rhostar_l + rhostar_r)
           un_state = ustar
           ut_state = 0.5*(ut_l + ut_r)
           p_state = pstar

        endif

        ! species now
        if (nspec > 0) then
           if (ustar > 0.0) then
              xn(:) = q_l(i,j,iX:iX-1+nspec)

           else if (ustar < 0.0) then
              xn(:) = q_r(i,j,iX:iX-1+nspec)
           else
              xn(:) = 0.5d0*(q_l(i,j,iX:iX-1+nspec) + q_r(i,j,iX:iX-1+nspec))
           endif
        endif

        ! are we on a solid boundary?
        if (idir == 1) then
           if (i == ilo .and. lower_solid == 1) then
              un_state = 0.0
           endif

           if (i == ihi+1 .and. upper_solid == 1) then
              un_state = 0.0
           endif

        else if (idir == 2) then
           if (j == jlo .and. lower_solid == 1) then
              un_state = 0.0
           endif

           if (j == jhi+1 .and. upper_solid == 1) then
              un_state = 0.0
           endif

        endif

        q_int(i,j,irho) = rho_state
        if (idir == 1) then
           q_int(i,j,iu) = un_state
           q_int(i,j,iv) = ut_state
        else
           q_int(i,j,iu) = ut_state
           q_int(i,j,iv) = un_state
        endif
        q_int(i,j,ip) = p_state

        if (nspec > 0) then
           q_int(i,j,iX:iX-1+nspec) = xn(:)
        endif

     enddo
  enddo

end subroutine riemann_prim


subroutine riemann_HLLC(idir, qx, qy, ng, &
                        nvar, idens, ixmom, iymom, iener, irhoX, irho, iu, iv, ip, ix, nspec, &
                        lower_solid, upper_solid, &
                        gamma, U_l, U_r, q_l, q_r, F)


  implicit none

  integer, intent(in) :: idir
  integer, intent(in) :: qx, qy, ng
  integer, intent(in) :: nvar, idens, ixmom, iymom, iener, irhoX, irho, iu, iv, ip, ix, nspec
  integer, intent(in) :: lower_solid, upper_solid
  double precision, intent(in) :: gamma

  ! 0-based indexing to match python
  double precision, intent(inout) :: U_l(0:qx-1,0:qy-1,0:nvar-1)
  double precision, intent(inout) :: U_r(0:qx-1,0:qy-1,0:nvar-1)
  double precision, intent(inout) :: q_l(0:qx-1,0:qy-1,0:nvar-1)
  double precision, intent(inout) :: q_r(0:qx-1,0:qy-1,0:nvar-1)
  double precision, intent(  out) :: F(0:qx-1,0:qy-1,0:nvar-1)

!f2py depend(qx, qy, nvar) :: q_l, q_r, U_l, U_r
!f2py intent(in) :: q_l, q_r, U_l, U_r
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
  double precision :: xn(nspec)

  double precision :: S_HLLE, E_HLLE, F_S, a_star, A, B
  double precision :: p_lstar, p_rstar, a_l, a_r, c_l, c_r
  double precision :: U_HLLE(0:nvar-1)
  double precision :: F_HLLE(0:nvar-1)
  double precision :: F_l(0:nvar-1)
  double precision :: F_r(0:nvar-1)
  double precision :: U_lstar(0:nvar-1)
  double precision :: U_rstar(0:nvar-1)

  double precision :: U_state(0:nvar-1)
  double precision :: q_state(0:nvar-1)



  nx = qx - 2*ng; ny = qy - 2*ng
  ilo = ng; ihi = ng+nx-1; jlo = ng; jhi = ng+ny-1

  do j = jlo-1, jhi+1
     do i = ilo-1, ihi+1

        ! primitive variable states
        ! rho_l  = q_l(i,j,irho)

        ! un = normal velocity; ut = transverse velocity
        if (idir == 1) then
           un_l    = q_l(i,j,iu)
           ut_l    = q_l(i,j,iv)
        else
           un_l    = q_l(i,j,iv)
           ut_l    = q_l(i,j,iu)
        endif

        ! rhoe_l = q_l(i,j,ip) / (gamma-1.0d0)

        p_l   = q_l(i,j,ip)
        p_l = max(p_l, smallp)

        ! rho_r  = q_r(i,j,irho)

        if (idir == 1) then
           un_r    = q_r(i,j,iu)
           ut_r    = q_r(i,j,iv)
        else
           un_r    = q_r(i,j,iv)
           ut_r    = q_r(i,j,iu)
        endif

        ! rhoe_r = q_r(i,j,ip) / (gamma-1.0d0)

        p_r   = q_r(i,j,ip)
        p_r = max(p_r, smallp)


        ! compute the sound speeds
        c_l = max(smallc, sqrt(gamma*p_l/rho_l))
        c_r = max(smallc, sqrt(gamma*p_r/rho_r))


        a_l = 0.5d0*(un_l+un_r - c_l-c_r) / (1.0d0-0.25d0*(un_l+un_r)*(c_l+c_r))
        a_r = 0.5d0*(un_l+un_r + c_l+c_r) / (1.0d0+0.25d0*(un_l+un_r)*(c_l+c_r))

        a_l = -1.0d0
        a_r = 1.0d0

        call consFlux(idir, gamma, idens, ixmom, iymom, iener, irhoX, iu, iv, ip, nvar, nspec, &
                      U_l(i,j,:), q_l(i,j,:), F_l)

        call consFlux(idir, gamma, idens, ixmom, iymom, iener, irhoX, iu, iv, ip, nvar, nspec, &
                    U_r(i,j,:), q_r(i,j,:), F_r)

        F_HLLE = (a_r*F_l - a_l*F_r + a_r*a_l*(U_r(i,j,:) - U_l(i,j,:))) / (a_r - a_l)

        ! q_star = (a_r*q_r(i,j,:) - a_l*q_l(i,j,:)-F_r+F_l) / (a_r - a_l)

        if (a_r <= 0.0d0) then ! right state
            U_HLLE = U_r(i,j,:)
        else if (a_l < 0.0d0) then !middle
            U_HLLE = (a_r*U_r(i,j,:) - a_l*U_l(i,j,:)-F_r+F_l) / (a_r - a_l)
        else ! left
            U_HLLE = U_l(i,j,:)
        endif

        if (idir == 1) then
            S_HLLE = U_HLLE(ixmom)
            F_S = F_HLLE(ixmom)
        else
            S_HLLE = U_HLLE(iymom)
            F_S = F_HLLE(iymom)
        endif

        E_HLLE = U_HLLE(iener) + U_HLLE(idens)

        if (abs(F_HLLE(iener)) < 1.d-9) then
            a_star = S_HLLE / (E_HLLE + F_S)
        else
            a_star = ( E_HLLE + F_S - &
                sqrt( (E_HLLE + F_S)**2 -  &
                S_HLLE * 2.0d0 * F_HLLE(iener) ) ) / &
                (2.0d0 * F_HLLE(iener))
        endif


        ! NOTE: this shouldn't happpen but just in case?
        if (a_star /= a_star) then
            a_star = 0.0d0
        endif

        ! left

        if (idir == 1) then
            A = (U_l(i,j,iener) + U_l(i,j,idens)) * a_l - U_l(i,j,ixmom)
            B = U_l(i,j,ixmom) * (a_l - un_l) - p_l
        else
            A = (U_l(i,j,iener) + U_l(i,j,idens)) * a_l - U_l(i,j,iymom)
            B = U_l(i,j,iymom) * (a_l - un_l) - p_l
        endif

        p_lstar = ((A * a_star) - B) / (1.0d0 - a_l * a_star)

        U_lstar(idens) = U_l(i,j,idens) * (a_l - un_l) / (a_l - a_star)

        if (idir == 1) then
            U_lstar(ixmom) = (U_l(i,j,ixmom) * (a_l - un_l) + p_lstar - p_l) / (a_l - a_star)
            U_lstar(iymom) = U_l(i,j,iymom) * (a_l - un_l) / (a_l - a_star)
        else
            U_lstar(ixmom) = U_l(i,j,ixmom) * (a_l - un_l)/ (a_l - a_star)
            U_lstar(iymom) = (U_l(i,j,iymom) * (a_l - un_l) + p_lstar - p_l) / (a_l - a_star)
        endif

        ! species
        if (nspec > 0) then
            U_lstar(irhoX:irhoX-1+nspec) = U_l(i,j,irhoX:irhoX-1+nspec) * (a_l - un_l)/ (a_l - a_star)
        endif


        ! right

        if (idir == 1) then
            A = (U_r(i,j,iener) + U_r(i,j,idens)) * a_r - U_r(i,j,ixmom)
            B = U_r(i,j,ixmom) * (a_r - un_r) - p_r
        else
            A = (U_r(i,j,iener) + U_r(i,j,idens)) * a_r - U_r(i,j,iymom)
            B = U_r(i,j,iymom) * (a_r - un_r) - p_r
        endif

        p_rstar = ((A * a_star) - B) / (1.0d0 - a_r * a_star)

        U_rstar(idens) = U_r(i,j,idens) * (a_r - un_r) / (a_r - a_star)

        if (idir == 1) then
            U_rstar(ixmom) = (U_r(i,j,ixmom) * (a_r - un_r) + p_rstar - p_r) / (a_r - a_star)
            U_rstar(iymom) = U_r(i,j,iymom) * (a_r - un_r) / (a_r - a_star)
        else
            U_rstar(ixmom) = U_r(i,j,ixmom) * (a_r - un_r)/ (a_r - a_star)
            U_rstar(iymom) = (U_r(i,j,iymom) * (a_r - un_r) + p_rstar - p_r) / (a_r - a_star)
        endif

        ! species
        if (nspec > 0) then
            U_rstar(irhoX:irhoX-1+nspec) = U_r(i,j,irhoX:irhoX-1+nspec) * (a_r - un_r)/ (a_r - a_star)
        endif

        if (a_r <= 0.0d0) then ! right state

            F(i,j,:) = F_r

        else if (a_star <= 0.0d0) then ! right star

            F(i,j,:) = U_rstar(:) * a_star

            if (idir == 1) then
                F(i,j,ixmom) = F(i,j,ixmom) + p_rstar
                F(i,j,iener) = U_rstar(ixmom) - F(i,j,idens)
            else
                F(i,j,iymom) = F(i,j,iymom) + p_rstar
                F(i,j,iener) = U_rstar(iymom) - F(i,j,idens)
            endif

        else if (a_l < 0.0d0) then ! left star

            F(i,j,:) = U_lstar(:) * a_star

            if (idir == 1) then
                F(i,j,ixmom) = F(i,j,ixmom) + p_lstar
                F(i,j,iener) = U_lstar(ixmom) - F(i,j,idens)
            else
                F(i,j,iymom) = F(i,j,iymom) + p_lstar
                F(i,j,iener) = U_lstar(iymom) - F(i,j,idens)
            endif

        else ! left

            F(i,j,:) = F_l

        endif

     enddo
  enddo

end subroutine riemann_HLLC

subroutine consFlux(idir, gamma, idens, ixmom, iymom, iener, irhoX, iu, iv, ip, nvar, nspec, U_state, q_state, F)

!f2py depend(nvar) :: U_state, q_state, F
!f2py intent(in) :: U_state, q_state
!f2py intent(out) :: F

  integer, intent(in) :: idir
  double precision, intent(in) :: gamma
  integer, intent(in) :: idens, ixmom, iymom, iener, irhoX, iu, iv, ip, nvar, nspec
  double precision, intent(in) :: U_state(0:nvar-1)
  double precision, intent(in) :: q_state(0:nvar-1)
  double precision, intent(out) :: F(0:nvar-1)

  double precision :: p, u, v

  u = q_state(iu)
  v = q_state(iv)

  p = q_state(ip)

  if (idir == 1) then
     F(idens) = U_state(idens)*u
     F(ixmom) = U_state(ixmom)*u + p
     F(iymom) = U_state(iymom)*u
     F(iener) = (U_state(iener) + p)*u
     if (nspec > 0) then
        F(irhoX:irhoX-1+nspec) = U_state(irhoX:irhoX-1+nspec)*u
     endif
  else
     F(idens) = U_state(idens)*v
     F(ixmom) = U_state(ixmom)*v
     F(iymom) = U_state(iymom)*v + p
     F(iener) = (U_state(iener) + p)*v
     if (nspec > 0) then
        F(irhoX:irhoX-1+nspec) = U_state(irhoX:irhoX-1+nspec)*v
     endif
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
