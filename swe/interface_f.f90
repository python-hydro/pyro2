subroutine states(idir, qx, qy, ng, dx, dt, &
                  ih, iu, iv, ix, nvar, nspec, &
                  g, &
                  qv, dqv, &
                  q_l, q_r)

  implicit none

  integer, intent(in) :: idir
  integer, intent(in) :: qx, qy, ng
  double precision, intent(in) :: dx, dt
  integer, intent(in) :: ih, iu, iv, ix, nvar, nspec
  double precision, intent(in) :: g

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
  !        / u   0   0 \
  !        | g   u   0 |
  !    A = \ 0   0   u /
  !
  ! The right eigenvectors are
  !
  !        /  h  \       /  0  \      /  h  \
  !   r1 = | -c  |  r2 = |  0  | r3 = |  c  |
  !        \  0  /       \  1  /      \  0  /
  !
  ! The left eigenvectors are
  !
  !    l1 =     ( 1/(2h),  -h/(2hc),  0 )
  !    l2 =     ( 0,          0,  1 )
  !    l3 =     ( -1/(2h), -h/(2hc),  0 )
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

  double precision :: dtdx, dtdx3
  double precision :: cs

  double precision :: sum, sum_l, sum_r, factor

  integer :: ns

  nx = qx - 2*ng; ny = qy - 2*ng
  ilo = ng; ihi = ng+nx-1; jlo = ng; jhi = ng+ny-1

  ns = nvar - nspec

  dtdx = dt/dx
  dtdx3 = 0.33333d0*dtdx

  ! this is the loop over zones.  For zone i, we see q_l[i+1] and q_r[i]
  do j = jlo-2, jhi+2
     do i = ilo-2, ihi+2

        dq(:) = dqv(i,j,:)
        q(:) = qv(i,j,:)

        cs = sqrt(g*q(ih))

        lvec(:,:) = 0.0d0
        rvec(:,:) = 0.0d0
        eval(:) = 0.0d0

        ! compute the eigenvalues and eigenvectors
        if (idir == 1) then
           eval(0:ns-1) = [q(iu) - cs, q(iu), q(iu) + cs]

           lvec(0,0:ns-1) = [ cs, -q(ih), 0.0d0 ]
           lvec(1,0:ns-1) = [ 0.0d0, 0.0d0, 1.0d0 ]
           lvec(2,0:ns-1) = [ cs, q(ih), 0.0d0 ]

           rvec(0,0:ns-1) = [ q(ih), -cs, 0.0d0 ]
           rvec(1,0:ns-1) = [ 0.0d0, 0.0d0, 1.0d0 ]
           rvec(2,0:ns-1) = [ q(ih), cs, 0.0d0 ]

           ! now the species -- they only have a 1 in their corresponding slot
           eval(ns:) = q(iu)
           do n = ix, ix-1+nspec
              lvec(n,n) = 1.0
              rvec(n,n) = 1.0
           enddo

           ! multiply by scaling factors
           lvec(0,:) = lvec(0,:) * 0.50d0 / (cs * q(ih))
           lvec(2,:) = -lvec(2,:) * 0.50d0 / (cs * q(ih))
        else
           eval(0:ns-1) = [q(iv) - cs, q(iv), q(iv) + cs]

           lvec(0,0:ns-1) = [ cs, 0.0d0, -q(ih) ]
           lvec(1,0:ns-1) = [ 0.0d0, 1.0d0, 0.0d0 ]
           lvec(2,0:ns-1) = [ cs, 0.0d0, q(ih) ]

           rvec(0,0:ns-1) = [ q(ih), 0.0d0, -cs ]
           rvec(1,0:ns-1) = [ 0.0d0, 1.0d0, 0.0d0 ]
           rvec(2,0:ns-1) = [ q(ih), 0.0d0, cs ]

           ! now the species -- they only have a 1 in their corresponding slot
           eval(ns:) = q(iv)
           do n = ix, ix-1+nspec
              lvec(n,n) = 1.0
              rvec(n,n) = 1.0
           enddo

           ! multiply by scaling factors
           lvec(0,:) = lvec(0,:) * 0.50d0 / (cs * q(ih))
           lvec(2,:) = -lvec(2,:) * 0.50d0 / (cs * q(ih))

        endif

        ! define the reference states
        if (idir == 1) then
           ! this is one the right face of the current zone,
           ! so the fastest moving eigenvalue is eval[2] = u + c
           factor = 0.5d0*(1.0d0 - dtdx*max(eval(2), 0.0d0))
           q_l(i+1,j,:) = q(:) + factor*dq(:)

           ! left face of the current zone, so the fastest moving
           ! eigenvalue is eval[3] = u - c
           factor = 0.5d0*(1.0d0 + dtdx*min(eval(0), 0.0d0))
           q_r(i,  j,:) = q(:) - factor*dq(:)

        else

           factor = 0.5d0*(1.0d0 - dtdx*max(eval(2), 0.0d0))
           q_l(i,j+1,:) = q(:) + factor*dq(:)

           factor = 0.5d0*(1.0d0 + dtdx*min(eval(0), 0.0d0))
           q_r(i,j,  :) = q(:) - factor*dq(:)

        endif

        ! compute the Vhat functions
        do m = 0, nvar-1
           sum = dot_product(lvec(m,:),dq(:))

           betal(m) = dtdx3*(eval(2) - eval(m))*(sign(1.0d0,eval(m)) + 1.0d0)*sum
           betar(m) = dtdx3*(eval(0) - eval(m))*(1.0d0 - sign(1.0d0,eval(m)))*sum
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


subroutine riemann_Roe(idir, qx, qy, ng, &
                        nvar, ih, ixmom, iymom, ihX, nspec, &
                        lower_solid, upper_solid, &
                        g, U_l, U_r, F)

  implicit none

  integer, intent(in) :: idir
  integer, intent(in) :: qx, qy, ng
  integer, intent(in) :: nvar, ih, ixmom, iymom, ihX, nspec
  integer, intent(in) :: lower_solid, upper_solid
  double precision, intent(in) :: g

  ! 0-based indexing to match python
  double precision, intent(inout) :: U_l(0:qx-1,0:qy-1,0:nvar-1)
  double precision, intent(inout) :: U_r(0:qx-1,0:qy-1,0:nvar-1)
  double precision, intent(  out) :: F(0:qx-1,0:qy-1,0:nvar-1)

!f2py depend(qx, qy, nvar) :: U_l, U_r
!f2py intent(in) :: U_l, U_r
!f2py intent(out) :: F

  ! This is the Roe Riemann solver with entropy fix. The implementation
  ! follows Toro's SWE book and the clawpack 2d SWE Roe solver.

  integer :: ilo, ihi, jlo, jhi
  integer :: nx, ny
  integer :: i, j, n, m
  integer :: ns

  double precision, parameter :: smallc = 1.e-10
  double precision, parameter :: tol = 0.1d-1 ! entropy fix parameter
  ! Note that I've basically assumed that cfl = 0.1 here to get away with
  ! not passing dx/dt or cfl to this function. If this isn't the case, will need
  ! to pass one of these to the function or else things will go wrong.

  double precision :: h_l, un_l, ut_l, c_l
  double precision :: h_r, un_r, ut_r, c_r
  double precision :: h_star, u_star, c_star
  double precision :: xn(nspec)

  double precision :: U_roe(0:nvar-1), c_roe, un_roe
  double precision :: lambda_roe(0:nvar-1), K_roe(0:nvar-1, 0:nvar-1)
  double precision :: alpha_roe(0:nvar-1), delta(0:nvar-1), F_r(0:nvar-1)

  nx = qx - 2*ng; ny = qy - 2*ng
  ilo = ng; ihi = ng+nx-1; jlo = ng; jhi = ng+ny-1
  ns = nvar - nspec

  do j = jlo-1, jhi+1
     do i = ilo-1, ihi+1

        ! primitive variable states
        h_l  = U_l(i,j,ih)

        ! un = normal velocity; ut = transverse velocity
        if (idir == 1) then
           un_l    = U_l(i,j,ixmom)/h_l
           ut_l    = U_l(i,j,iymom)/h_l
        else
           un_l    = U_l(i,j,iymom)/h_l
           ut_l    = U_l(i,j,ixmom)/h_l
        endif

        h_r  = U_r(i,j,ih)

        if (idir == 1) then
           un_r    = U_r(i,j,ixmom)/h_r
           ut_r    = U_r(i,j,iymom)/h_r
        else
           un_r    = U_r(i,j,iymom)/h_r
           ut_r    = U_r(i,j,ixmom)/h_r
        endif

        ! compute the sound speeds
        c_l = max(smallc, sqrt(g*h_l))
        c_r = max(smallc, sqrt(g*h_r))

        ! Calculate the Roe averages
        U_roe = (U_l(i,j,:)/sqrt(h_l) + U_r(i,j,:)/sqrt(h_r)) / &
                  (sqrt(h_l) + sqrt(h_r))

        U_roe(ih) = sqrt(h_l * h_r)
        c_roe = sqrt(0.5d0 * (c_l**2 + c_r**2))

        delta(:) = U_r(i,j,:)/h_r - U_l(i,j,:)/h_l
        delta(ih) = h_r - h_l

        ! evalues and right evectors
        if (idir == 1) then
          un_roe = U_roe(ixmom)
        else
          un_roe = U_roe(iymom)
        endif

        K_roe(:,:) = 0.0d0

        lambda_roe(0:2) = [un_roe - c_roe, un_roe, un_roe + c_roe]
        if (idir == 1) then
          alpha_roe(0:2) = [0.5d0*(delta(ih) - U_roe(ih)/c_roe*delta(ixmom)), &
                           U_roe(ih) * delta(iymom), &
                           0.5d0*(delta(ih) + U_roe(ih)/c_roe*delta(ixmom))]

          K_roe(0, 0:2) = [1.0d0, un_roe - c_roe, U_roe(iymom)]
          K_roe(1, 0:2) = [0.0d0, 0.0d0, 1.0d0]
          K_roe(2, 0:2) = [1.0d0, un_roe + c_roe, U_roe(iymom)]
        else
          alpha_roe(0:2) = [0.5d0*(delta(ih) - U_roe(ih)/c_roe*delta(iymom)), &
                           U_roe(ih) * delta(ixmom), &
                           0.5d0*(delta(ih) + U_roe(ih)/c_roe*delta(iymom))]

          K_roe(0, 0:2) = [1.0d0, U_roe(ixmom), un_roe - c_roe]
          K_roe(1, 0:2) = [0.0d0, 1.0d0, 0.0d0]
          K_roe(2, 0:2) = [1.0d0, U_roe(ixmom), un_roe + c_roe]
        endif

        lambda_roe(ns:) = un_roe
        alpha_roe(ns:) = U_roe(ih) * delta(ns:)
        do n = ns, nvar-1
           K_roe(n,:) = 0.0d0
           K_roe(n,n) = 1.0d0
        enddo

        call consFlux(idir, g, ih, ixmom, iymom, ihX, nvar, nspec, &
                      U_l(i,j,:), F(i,j,:))
        call consFlux(idir, g, ih, ixmom, iymom, ihX, nvar, nspec, &
                      U_r(i,j,:), F_r)

        F(i,j,:) = 0.5d0 * (F(i,j,:) + F_r)

        h_star = 1.0d0 / g * (0.5d0 * (c_l + c_r) + 0.25d0 * (un_l - un_r))**2
        u_star = 0.5d0 * (un_l + un_r) + c_l - c_r

        c_star = sqrt(g * h_star)

        ! modified evalues for entropy fix
        if (abs(lambda_roe(0)) < tol) then
          lambda_roe(0) = lambda_roe(0) * (u_star - c_star - lambda_roe(0)) / &
            (u_star - c_star - (un_l - c_l))
        endif
        if (abs(lambda_roe(2)) < tol) then
          lambda_roe(2) = lambda_roe(2) * (u_star + c_star - lambda_roe(2)) / &
            (u_star + c_star - (un_r + c_r))
        endif

        do n = 0, nvar-1
          do m = 0, nvar-1
            F(i,j,n) = F(i,j,n) - &
                0.5d0 * alpha_roe(m) * abs(lambda_roe(m)) * K_roe(m,n)
          enddo
        enddo

     enddo
  enddo
end subroutine riemann_Roe


subroutine riemann_HLLC(idir, qx, qy, ng, &
                        nvar, ih, ixmom, iymom, ihX, nspec, &
                        lower_solid, upper_solid, &
                        g, U_l, U_r, F)

  implicit none

  integer, intent(in) :: idir
  integer, intent(in) :: qx, qy, ng
  integer, intent(in) :: nvar, ih, ixmom, iymom, ihX, nspec
  integer, intent(in) :: lower_solid, upper_solid
  double precision, intent(in) :: g

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

  double precision :: h_l, un_l, ut_l
  double precision :: h_r, un_r, ut_r
  double precision :: xn(nspec)

  double precision :: hstar_l, hstar_r, h_avg
  double precision :: hstar, ustar, u_avg
  double precision :: S_l, S_r, S_c
  double precision :: c_l, c_r, c_avg

  double precision :: U_state(0:nvar-1)
  double precision :: HLLCfactor

  nx = qx - 2*ng; ny = qy - 2*ng
  ilo = ng; ihi = ng+nx-1; jlo = ng; jhi = ng+ny-1

  do j = jlo-1, jhi+1
     do i = ilo-1, ihi+1

        ! primitive variable states
        h_l  = U_l(i,j,ih)

        ! un = normal velocity; ut = transverse velocity
        if (idir == 1) then
           un_l    = U_l(i,j,ixmom)/h_l
           ut_l    = U_l(i,j,iymom)/h_l
        else
           un_l    = U_l(i,j,iymom)/h_l
           ut_l    = U_l(i,j,ixmom)/h_l
        endif

        h_r  = U_r(i,j,ih)

        if (idir == 1) then
           un_r    = U_r(i,j,ixmom)/h_r
           ut_r    = U_r(i,j,iymom)/h_r
        else
           un_r    = U_r(i,j,iymom)/h_r
           ut_r    = U_r(i,j,ixmom)/h_r
        endif

        ! compute the sound speeds
        c_l = max(smallc, sqrt(g*h_l))
        c_r = max(smallc, sqrt(g*h_r))

        ! Estimate the star quantities -- use one of three methods to
        ! do this -- the primitive variable Riemann solver, the two
        ! shock approximation, or the two rarefaction approximation.
        ! Pick the method based on the pressure states at the
        ! interface.

        h_avg = 0.5*(h_l + h_r)
        c_avg = 0.5*(c_l + c_r)
        u_avg = 0.5*(un_l + un_r)

        hstar = h_avg - 0.25d0 * (un_r - un_l) * h_avg / c_avg
        ustar = u_avg - (h_r - h_l) * c_avg / h_avg

        ! estimate the nonlinear wave speeds

        if (hstar <= h_l) then
           ! rarefaction
           S_l = un_l - c_l
        else
           ! shock
           S_l = un_l - c_l * sqrt(0.5d0 * (hstar+h_l) * hstar) / h_l
        endif

        if (hstar <= h_r) then
           ! rarefaction
           S_r = un_r + c_r
        else
           ! shock
           S_r = un_r + c_r*sqrt(0.5d0 * (hstar+h_r) * hstar) / h_r
        endif

        S_c = (S_l*h_r*(un_r-S_r) - S_r*h_l*(un_l-S_l)) / &
                (h_r*(un_r-S_r) - h_l*(un_l-S_l))

        ! figure out which region we are in and compute the state and
        ! the interface fluxes using the HLLC Riemann solver
        if (S_r <= 0.0d0) then
           ! R region
           U_state(:) = U_r(i,j,:)

           call consFlux(idir, g, ih, ixmom, iymom, ihX, nvar, nspec, &
                         U_state, F(i,j,:))

        else if (S_r > 0.0d0 .and. S_c <= 0) then
           ! R* region
           HLLCfactor = h_r*(S_r - un_r)/(S_r - S_c)

           U_state(ih) = HLLCfactor

           if (idir == 1) then
              U_state(ixmom) = HLLCfactor*S_c
              U_state(iymom) = HLLCfactor*ut_r
           else
              U_state(ixmom) = HLLCfactor*ut_r
              U_state(iymom) = HLLCfactor*S_c
           endif

           ! species
           if (nspec > 0) then
              U_state(ihX:ihX-1+nspec) = HLLCfactor*U_r(i,j,ihX:ihX-1+nspec)/h_r
           endif

           ! find the flux on the right interface
           call consFlux(idir, g, ih, ixmom, iymom, ihX, nvar, nspec, &
                         U_r(i,j,:), F(i,j,:))

           ! correct the flux
           F(i,j,:) = F(i,j,:) + S_r*(U_state(:) - U_r(i,j,:))

        else if (S_c > 0.0d0 .and. S_l < 0.0) then
           ! L* region
           HLLCfactor = h_l*(S_l - un_l)/(S_l - S_c)

           U_state(ih) = HLLCfactor

           if (idir == 1) then
              U_state(ixmom) = HLLCfactor*S_c
              U_state(iymom) = HLLCfactor*ut_l
           else
              U_state(ixmom) = HLLCfactor*ut_l
              U_state(iymom) = HLLCfactor*S_c
           endif

           ! species
           if (nspec > 0) then
              U_state(ihX:ihX-1+nspec) = HLLCfactor*U_l(i,j,ihX:ihX-1+nspec)/h_l
           endif

           ! find the flux on the left interface
           call consFlux(idir, g, ih, ixmom, iymom, ihX, nvar, nspec, &
                         U_l(i,j,:), F(i,j,:))

           ! correct the flux
           F(i,j,:) = F(i,j,:) + S_l*(U_state(:) - U_l(i,j,:))

        else
           ! L region
           U_state(:) = U_l(i,j,:)

           call consFlux(idir, g, ih, ixmom, iymom, ihX, nvar, nspec, &
                         U_state, F(i,j,:))

        endif

        ! we should deal with solid boundaries somehow here

     enddo
  enddo
end subroutine riemann_HLLC


subroutine consFlux(idir, g, ih, ixmom, iymom, ihX, nvar, nspec, U_state, F)

  implicit none

  integer, intent(in) :: idir
  double precision, intent(in) :: g
  integer, intent(in) :: ih, ixmom, iymom, ihX, nvar, nspec
  double precision, intent(in) :: U_state(0:nvar-1)
  double precision, intent(out) :: F(0:nvar-1)

! Calculate the conserved flux for the shallow water equations. In the
! x-direction, this is given by
!
!     /      hu       \
! F = | hu^2 + gh^2/2 |
!     \      huv      /

  double precision :: u, v

  u = U_state(ixmom)/U_state(ih)
  v = U_state(iymom)/U_state(ih)

  if (idir == 1) then
     F(ih) = U_state(ih)*u
     F(ixmom) = U_state(ixmom)*u + 0.5d0 * g * U_state(ih)**2
     F(iymom) = U_state(iymom)*u
     if (nspec > 0) then
        F(ihX:ihX-1+nspec) = U_state(ihX:ihX-1+nspec)*u
     endif
  else
     F(ih) = U_state(ih)*v
     F(ixmom) = U_state(ixmom)*v
     F(iymom) = U_state(iymom)*v + 0.5d0 * g * U_state(ih)**2
     if (nspec > 0) then
        F(ihX:ihX-1+nspec) = U_state(ihX:ihX-1+nspec)*v
     endif
  endif

end subroutine consFlux
