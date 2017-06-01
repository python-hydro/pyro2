! this library implements the limiting functions used in the
! reconstruction.  We use F90 for both speed and clarity, since
! the numpy array notations can sometimes be confusing.


!-----------------------------------------------------------------------------
! multi-dimensional limiting from BDS
!-----------------------------------------------------------------------------
subroutine multid_limit(a, ldax, lday, qx, qy, ng)

  ! a multidimensional limiter based on the ideas from
  ! Bell, Dawson, and Schubin

  implicit none

  integer, intent(in) :: qx, qy, ng

  ! 0-based indexing to match python
  double precision, intent(inout) :: a(0:qx-1, 0:qy-1)
  double precision, intent(  out) :: ldax(0:qx-1, 0:qy-1), lday(0:qx-1, 0:qy-1)

!f2py depend(qx, qy) :: a, ldax, lday
!f2py intent(in) :: a
!f2py intent(out) :: ldax, lday

! call this as: ldax, lday = reconstruction_f.multid_limit(a,qx,qy,ng)

  integer :: ilo, ihi, jlo, jhi
  integer :: nx, ny
  integer :: i, j, m, n

  integer, parameter :: ill = 1, ilh = 2, irl = 3, irh = 4
  double precision :: ss(4), ss_temp(4), min_ss(4), max_ss(4), diff(4)
  double precision :: A_x, A_y, A_xy
  double precision :: sumdiff, sgndiff, redfac, redmax, div
  double precision, parameter :: eps = 1.d-10
  integer :: kdp
  integer, parameter :: niter = 3

  double precision, allocatable :: a_nd(:,:)

  nx = qx - 2*ng; ny = qy - 2*ng
  ilo = ng; ihi = ng+nx-1; jlo = ng; jhi = ng+ny-1

  ldax(:,:) = 0.0d0
  lday(:,:) = 0.0d0

  ! here we do a multidimensional reconstruction of the data, forming
  ! a bilinear polynomial: 
  !     q(x,y) = A_xy (x-x_i)(y-y_i) + A_x (x-x_i) + A_y (y-y_i) + A_avg
  !
  ! First find the values at the nodes.  Here a_nd(i,j) will refer to
  ! a_{i-1/2,j-1/2}

  allocate(a_nd(0:qx-1, 0:qy-1))

  do j = jlo-2, jhi+3
     do i = ilo-2, ihi+3
          
        a_nd(i,j) = &
             (        a(i-2,j-2) -  7.0d0*(a(i-1,j-2) + a(i,j-2)) +       a(i+1,j-2) + &
             (-7.0d0)*a(i-2,j-1) + 49.0d0*(a(i-1,j-1) + a(i,j-1)) - 7.0d0*a(i+1,j-1) + &
             (-7.0d0)*a(i-2,j  ) + 49.0d0*(a(i-1,j  ) + a(i,j  )) - 7.0d0*a(i+1,j  ) + &
                      a(i-2,j+1) -  7.0d0*(a(i-1,j+1) + a(i,j+1)) +       a(i+1,j+1))/ &
             144.0d0

     enddo
  enddo

  ! now get the coefficients of the bilinear polynomial.  Here we loop
  ! ! over zone centers.
  do j = jlo-1, jhi+1
     do i = ilo-1, ihi+1
          
        ss(ill) = a_nd(i,j)     ! a_{i-1/2,j-1/2}
        ss(ilh) = a_nd(i,j+1)   ! a_{i-1/2,j+1/2}
        ss(irl) = a_nd(i+1,j)   ! a_{i+1/2,j-1/2}
        ss(irh) = a_nd(i+1,j+1) ! a_{i+1/2,j+1/2}
          
        ! note, we omit the dx and dy here
        A_x  = ((ss(irh) + ss(irl)) - (ss(ilh) + ss(ill)))/2.0d0
        A_y  = ((ss(ilh) + ss(irh)) - (ss(ill) + ss(irl)))/2.0d0
        A_xy = ((ss(irh) - ss(irl)) - (ss(ilh) - ss(ill)))
          
        ! check if we are within the cc-values
        ss_temp(ill) = a(i,j) - 0.5d0*A_x - 0.5d0*A_y + 0.25d0*A_xy
        ss_temp(ilh) = a(i,j) - 0.5d0*A_x + 0.5d0*A_y - 0.25d0*A_xy
        ss_temp(irl) = a(i,j) + 0.5d0*A_x - 0.5d0*A_y - 0.25d0*A_xy
        ss_temp(irh) = a(i,j) + 0.5d0*A_x + 0.5d0*A_y + 0.25d0*A_xy
          
        min_ss(ill) = min(a(i-1,j-1), a(i,j-1), a(i-1,j), a(i,j))
        max_ss(ill) = max(a(i-1,j-1), a(i,j-1), a(i-1,j), a(i,j))
          
        min_ss(ilh) = min(a(i-1,j+1), a(i,j+1), a(i-1,j), a(i,j))
        max_ss(ilh) = max(a(i-1,j+1), a(i,j+1), a(i-1,j), a(i,j))
          
        min_ss(irl) = min(a(i+1,j-1), a(i,j-1), a(i+1,j), a(i,j))
        max_ss(irl) = max(a(i+1,j-1), a(i,j-1), a(i+1,j), a(i,j))
        
        min_ss(irh) = min(a(i+1,j+1), a(i,j+1), a(i+1,j), a(i,j))
        max_ss(irh) = max(a(i+1,j+1), a(i,j+1), a(i+1,j), a(i,j))
          
        ! limit
        do m = 1, 4
           ss_temp(m) = max(min(ss_temp(m), max_ss(m)), min_ss(m))
        enddo

        do n = 1, niter
           sumdiff = (ss_temp(ill) + ss_temp(ilh) + &
                      ss_temp(irl) + ss_temp(irh)) - 4.0d0*a(i,j)
           sgndiff = sign(1.d0, sumdiff)

           do m = 1, 4
              diff(m) = (ss_temp(m) - a(i,j))*sgndiff
           enddo
             
           kdp = 0
             
           do m = 1, 4
              if (diff(m) > eps) kdp = kdp + 1
           enddo
             
           do m = 1, 4
              if (kdp < 1) then
                 div = 1.d0
              else
                 div = dble(kdp)
              endif
                
              if (diff(m) > eps) then
                 redfac = sumdiff*sgndiff/div
                 kdp = kdp-1
              else
                 redfac = 0.0d0
              endif
              
              if (sgndiff > 0.0d0) then
                 redmax = ss_temp(m) - min_ss(m)
              else
                 redmax = max_ss(m) - ss_temp(m)
              endif
                
              redfac = min(redfac, redmax)
              sumdiff = sumdiff - redfac*sgndiff
              ss_temp(m) = ss_temp(m) - redfac*sgndiff
           enddo
        enddo
          
        ! construct the final slopes
        A_x  = ((ss_temp(irh) + ss_temp(irl)) - &
                (ss_temp(ilh) + ss_temp(ill)))/2.0d0

        A_y  = ((ss_temp(ilh) + ss_temp(irh)) - &
                (ss_temp(ill) + ss_temp(irl)))/2.0d0

        !A_xy = ((ss_temp(irh) - ss_temp(irl)) - &
        !        (ss_temp(ilh) - ss_temp(ill)))/(dx*dy)

        ! now construct the limited differences in x and y
        ldax(i,j) = A_x
        lday(i,j) = A_y

     enddo
  enddo

  deallocate(a_nd)

end subroutine multid_limit


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
