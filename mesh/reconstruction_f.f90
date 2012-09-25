! this library implements the limiting functions used in the
! reconstruction.  We use F90 for both speed and clarity, since
! the numpy array notations can sometimes be confusing.

!-----------------------------------------------------------------------------
! nolimit
!-----------------------------------------------------------------------------
subroutine nolimit(idir, a, lda, qx, qy, ng)

  ! just to a centered difference -- no limiting
  implicit none

  integer, intent(in) :: idir
  integer, intent(in) :: qx, qy, ng

  ! 0-based indexing to match python
  double precision, intent(inout) :: a(0:qx-1, 0:qy-1)
  double precision, intent(  out) :: lda(0:qx-1, 0:qy-1)

!f2py intent(in) :: idir
!f2py depend(qx, qy) :: a, lda
!f2py intent(in) :: a
!f2py intent(out) :: lda

! call this as: lda = reconstruction_f.nolimit(1,a,qx,qy,ng)

  integer :: ilo, ihi, jlo, jhi
  integer :: nx, ny
  integer :: i, j

  nx = qx - 2*ng; ny = qy - 2*ng
  ilo = ng; ihi = ng+nx-1; jlo = ng; jhi = ng+ny-1

  lda(:,:) = 0.0d0

  select case (idir)

  case (1)

     do j = jlo-2, jhi+2
        do i = ilo-2, ihi+2
           lda(i,j) = 0.5d0*(a(i+1,j) - a(i-1,j))
        enddo
     enddo

  case (2)

     do j = jlo-2, jhi+2
        do i = ilo-2, ihi+2
           lda(i,j) = 0.5d0*(a(i,j+1) - a(i,j-1))
        enddo
     enddo

  end select

end subroutine nolimit


!-----------------------------------------------------------------------------
! limit2
!-----------------------------------------------------------------------------
subroutine limit2(idir, a, lda, qx, qy, ng)

  ! 2nd order limited centered difference
  implicit none

  integer, intent(in) :: idir
  integer, intent(in) :: qx, qy, ng

  ! 0-based indexing to match python
  double precision, intent(inout) :: a(0:qx-1, 0:qy-1)
  double precision, intent(  out) :: lda(0:qx-1, 0:qy-1)

!f2py intent(in) :: idir
!f2py depend(qx, qy) :: a, lda
!f2py intent(in) :: a
!f2py intent(out) :: lda

! call this as: lda = reconstruction_f.limit2(1,a,qx,qy,ng)

  integer :: ilo, ihi, jlo, jhi
  integer :: nx, ny
  integer :: i, j

  double precision :: test

  nx = qx - 2*ng; ny = qy - 2*ng
  ilo = ng; ihi = ng+nx-1; jlo = ng; jhi = ng+ny-1

  lda(:,:) = 0.0d0

  select case (idir)

  case (1)

     do j = jlo-3, jhi+3
        do i = ilo-3, ihi+3

           ! test whether we are at an extremum
           test = (a(i+1,j) - a(i,j))*(a(i,j) - a(i-1,j))

           if (test > 0.0d0) then
              lda(i,j) = min(0.5d0*abs(a(i+1,j) - a(i-1,j)), &
                             min(2.0d0*abs(a(i+1,j) - a(i,j)), &
                                 2.0d0*abs(a(i,j) - a(i-1,j)))) * &
                         sign(1.0d0,a(i+1,j) - a(i-1,j))

           else
              lda(i,j) = 0.0d0
           endif

        enddo
     enddo

  case (2)

     do j = jlo-3, jhi+3
        do i = ilo-3, ihi+3

           ! test whether we are at an extremum
           test = (a(i,j+1) - a(i,j))*(a(i,j) - a(i,j-1))

           if (test > 0.0d0) then
              lda(i,j) = min(0.5d0*abs(a(i,j+1) - a(i,j-1)), &
                             min(2.0d0*abs(a(i,j+1) - a(i,j)), &
                                 2.0d0*abs(a(i,j) - a(i,j-1)))) * &
                         sign(1.0d0,a(i,j+1) - a(i,j-1))

           else
              lda(i,j) = 0.0d0
           endif

        enddo
     enddo

  end select

end subroutine limit2


!-----------------------------------------------------------------------------
! limit4
!-----------------------------------------------------------------------------
subroutine limit4(idir, a, lda, qx, qy, ng)

  ! 4th order limited centered difference
  !
  ! See Colella (1985) Eq. 2.5 and 2.6, Colella (1990) page 191 (with
  ! the delta a terms all equal) or Saltzman 1994, page 156
  implicit none

  integer, intent(in) :: idir
  integer, intent(in) :: qx, qy, ng

  ! 0-based indexing to match python
  double precision, intent(inout) :: a(0:qx-1, 0:qy-1)
  double precision, intent(  out) :: lda(0:qx-1, 0:qy-1)

!f2py intent(in) :: idir
!f2py depend(qx, qy) :: a, lda
!f2py intent(in) :: a
!f2py intent(out) :: lda

! call this as: lda = reconstruction_f.limit4(1,a,qx,qy,ng)

  integer :: ilo, ihi, jlo, jhi
  integer :: nx, ny
  integer :: i, j

  double precision :: test
  double precision :: temp(0:qx-1,0:qy-1)

  nx = qx - 2*ng; ny = qy - 2*ng
  ilo = ng; ihi = ng+nx-1; jlo = ng; jhi = ng+ny-1

  lda(:,:) = 0.0d0

  ! first get the 2nd order estimate
  call limit2(idir, a, temp, qx, qy, ng)

  select case (idir)

  case (1)

     do j = jlo-2, jhi+2
        do i = ilo-2, ihi+2

           ! test whether we are at an extremum
           test = (a(i+1,j) - a(i,j))*(a(i,j) - a(i-1,j))

           if (test > 0.0d0) then
              lda(i,j) = &
                   min( (2.0d0/3.0d0)*abs(a(i+1,j) - a(i-1,j) - &
                         0.25d0*(temp(i+1,j) + temp(i-1,j))), &
                       min(2.0d0*abs(a(i+1,j) - a(i  ,j)), &
                           2.0d0*abs(a(i  ,j) - a(i-1,j))) ) * &
                   sign(1.0d0,a(i+1,j) - a(i-1,j))
              
           else
              lda(i,j) = 0.0d0
           endif

        enddo
     enddo

  case (2)

     do j = jlo-2, jhi+2
        do i = ilo-2, ihi+2

           ! test whether we are at an extremum
           test = (a(i,j+1) - a(i,j))*(a(i,j) - a(i,j-1))

           if (test > 0.0d0) then
              lda(i,j) = &
                   min( (2.0d0/3.0d0)*abs(a(i,j+1) - a(i,j-1) - &
                         0.25d0*(temp(i,j+1) + temp(i,j-1))), &
                       min(2.0d0*abs(a(i,j+1) - a(i,j  )), &
                           2.0d0*abs(a(i,j  ) - a(i,j-1))) ) * &
                   sign(1.0d0,a(i,j+1) - a(i,j-1))

           else
              lda(i,j) = 0.0d0
           endif

        enddo
     enddo

  end select

end subroutine limit4


!-----------------------------------------------------------------------------
! flatten
!-----------------------------------------------------------------------------
subroutine flatten(idir, p, u, xi, qx, qy, ng, smallp, delta, z0, z1)

  ! 1-d flattening near shocks
  !
  ! flattening kicks in behind strong shocks and reduces the
  ! reconstruction to using piecewise constant slopes, making things
  ! first-order.  See Saltzman (1994) page 159 for this
  ! implementation.
  implicit none

  integer, intent(in) :: idir
  integer, intent(in) :: qx, qy, ng

  ! 0-based indexing to match python
  double precision, intent(inout) :: p(0:qx-1, 0:qy-1), u(0:qx-1, 0:qy-1)
  double precision, intent(  out) :: xi(0:qx-1, 0:qy-1)
  double precision, intent(in   ) :: smallp, delta, z0, z1

!f2py intent(in) :: idir
!f2py depend(qx, qy) :: p, u, xi
!f2py intent(in) :: p, u
!f2py intent(out) :: xi

! call this as: xi = reconstruction_f.flatten(1,p,u,qx,qy,ng,smallp,delta,z0,z1)

  integer :: ilo, ihi, jlo, jhi
  integer :: nx, ny
  integer :: i, j

  double precision :: test1, test2
  double precision :: dp, dp2, z

  nx = qx - 2*ng; ny = qy - 2*ng
  ilo = ng; ihi = ng+nx-1; jlo = ng; jhi = ng+ny-1

  xi(:,:) = 1.0d0

  select case (idir)

  case (1)

     do j = jlo-2, jhi+2
        do i = ilo-2, ihi+2

           dp  = abs(p(i+1,j) - p(i-1,j))
           dp2 = abs(p(i+2,j) - p(i-2,j))
           z = dp/max(dp2,smallp)

           test1 = u(i-1,j) - u(i+1,j)
           test2 = dp/min(p(i+1,j),p(i-1,j))

           if (test1 > 0.0d0 .and. test2 > delta) then
              xi(i,j) = min(1.0d0, max(0.0d0, 1.0d0 - (z - z0)/(z1 - z0)))
           else
              xi(i,j) = 1.0
           endif

        enddo
     enddo

  case (2)

     do j = jlo-2, jhi+2
        do i = ilo-2, ihi+2

           dp  = abs(p(i,j+1) - p(i,j-1))
           dp2 = abs(p(i,j+2) - p(i,j-2))
           z = dp/max(dp2,smallp)

           test1 = u(i,j-1) - u(i,j+1)
           test2 = dp/min(p(i,j+1),p(i,j-1))

           if (test1 > 0.0d0 .and. test2 > delta) then
              xi(i,j) = min(1.0d0, max(0.0d0, 1.0d0 - (z - z0)/(z1 - z0)))
           else
              xi(i,j) = 1.0
           endif

        enddo
     enddo

  end select

end subroutine flatten

    
!-----------------------------------------------------------------------------
! flatten_multid
!-----------------------------------------------------------------------------
subroutine flatten_multid(xi_x, xi_y, p, xi, qx, qy, ng)

  ! multi-dimensional flattening
  implicit none

  integer, intent(in) :: qx, qy, ng

  ! 0-based indexing to match python
  double precision, intent(inout) :: xi_x(0:qx-1, 0:qy-1)
  double precision, intent(inout) :: xi_y(0:qx-1, 0:qy-1)
  double precision, intent(inout) :: p(0:qx-1, 0:qy-1)
  double precision, intent(  out) :: xi(0:qx-1, 0:qy-1)

!f2py depend(qx, qy) :: xi_x, xi_y, p, xi
!f2py intent(in) :: xi_x, xi_y, p
!f2py intent(out) :: xi

! call this as: xi = reconstruction_f.flatten_multid(xi_x,xi_y,,p,qx,qy,ng)

  integer :: ilo, ihi, jlo, jhi
  integer :: nx, ny
  integer :: i, j

  integer :: sx, sy

  nx = qx - 2*ng; ny = qy - 2*ng
  ilo = ng; ihi = ng+nx-1; jlo = ng; jhi = ng+ny-1

  xi(:,:) = 1.0d0

  do j = jlo-2, jhi+2
     do i = ilo-2, ihi+2

        sx = int(sign(1.0d0, p(i+1,j) - p(i-1,j)))
        sy = int(sign(1.0d0, p(i,j+1) - p(i,j-1)))

        xi(i,j) = min(min(xi_x(i,j), xi_x(i-sx,j)), &
                      min(xi_y(i,j), xi_y(i,j-sy)))

     enddo
  enddo

end subroutine flatten_multid
