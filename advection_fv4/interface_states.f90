subroutine limit(a, qx, qy, ng, idir, &
                 al, ar)

  implicit none

  integer, intent(in) :: idir
  integer, intent(in) :: qx, qy, ng

  double precision, intent(inout) :: a(0:qx-1, 0:qy-1)

  double precision, intent(out) :: al(0:qx-1, 0:qy-1)
  double precision, intent(out) :: ar(0:qx-1, 0:qy-1)

  double precision :: aint(0:qx-1, 0:qy-1)
  double precision :: dafm(0:qx-1, 0:qy-1)
  double precision :: dafp(0:qx-1, 0:qy-1)
  double precision :: d2af(0:qx-1, 0:qy-1)
  double precision :: d2ac(0:qx-1, 0:qy-1)
  double precision :: d3a(0:qx-1, 0:qy-1)

  integer :: ilo, ihi, jlo, jhi
  integer :: nx, ny

  nx = qx - 2*ng; ny = qy - 2*ng
  ilo = ng; ihi = ng+nx-1; jlo = ng; jhi = ng+ny-1

  ! our convention here is that:
  !     al(i,j)   will be al_{i-1/2,j),
  !     al(i+1,j) will be al_{i+1/2,j)

  ! we need interface values on all faces of the domain
  if (idir == 1) then

     do j = jlo, jhi
        do i = ilo-2, ihi+3

           ! interpolate to the edges
           a_int(i,j) = (7.0/12.0)*(a(i-1,j) + a(i,j)) - (1.0/12.0)*(a(i-2,j) + a(i+1,j))

           al(i,j) = a_int(i,j)
           ar(i,j) = a_int(i,j)

        enddo
     enddo

     do j = jlo, jhi
        do i = ilo-1, ihi+2
           ! these live on cell-centers
           dafm(i,j) = a(i,j) - a_int(i,j)
           dafp(i,j) = a_int(i+1,j) - a(i,j)

           ! these live on cell-centers
           d2af(i,j) = 6.0*(a_int(i,j) - 2.0*a(i,j) + a_int(i+1,j))
           d2ac(i,j) = a(i-1,j) - 2.0*a(i,j) + a(i+1,j)

           ! this lives on the interface
           d3a(i,j) = d2ac(i,j) - d2ac(i-1,j)

        enddo
     enddo

     do j = jlo, jhi
        do i = ilo, ihi+1

           ! limit? MC Eq. 24 and 25
           if (dafm(i,j) * dafp(i,j) <= 0.0 .or. &
                (a(i,j) - a(i-2,j))*(a(i+2,j) - a(i,j)) <= 0.0) then

              ! we are at an extrema

              s = sign(1.0, d2ac(i,j))
              if (s == sign(1.0, d2ac(i-1,j)) .and. s == sign(1.0, d2ac(i+1,j)) .and. &
                   s == sign(1.0, d2af(i,j))) then
                 ! MC Eq. 26
                 d2a_lim = s*min(abs(d2af(i,j)), C2*abs(d2ac(i-1,j)), &
                                 C2*abs(d2ac(i,j)), C2*abs(d2ac(i+1,j)))
              else
                 d2a_lim = 0.0d0
              endif

              if (abs(d2af(i,j)) <= 1.e-12*max(abs(a(i-2,j)), abs(a(i-1,j)), &
                                               abs(a(i,j)), abs(a(i+1,j)), abs(a(i+2,j)))) then
                 rho = 0.0
              else
                 ! MC Eq. 27
                 rho = d2a_lim/d2af(i,j)
              enddo

              if (rho < 1.0d0 - 1.d-12) then
                 ! we may need to limit -- these quantities are at cell-centers
                 d3a_min = min(d3a(i-1,j), d3a(i,j), d3a(i+1,j), d3a(i+2,j))
                 d3a_max = max(d3a(i-1,j), d3a(i,j), d3a(i+1,j), d3a(i+2,j))

                 if (C3*max(abs(d3a_min), abs(d3a_max)) <= (d3a_max - d3a_min)) then
                    ! limit
                    if (dafm(i,j)*dafp(i,j) < 0.0d0) then
                       ! Eqs. 29, 30
                       ar(i,j) = a(i,j) - rho*dafm(i,j)  ! note: typo in Eq 29
                       al(i+1,j) = a(i,j) + rho*dafp(i,j)
                    else if (abs(dafm(i,j)) >= 2.0*abs(dafp(i,j))) then
                       ! Eq. 31
                       ar(i,j) = a(i,j) - 2.0d0*(1.0d0 - rho)*dafp(i,j) - rho*dafm(i,j)
                    else if (abs(dafp(i,j)) >= 2.0*abs(dafm(i,j))) then
                       ! Eq. 32
                       al(i+1,j) = a(i,j) + 2.0d0*(1.0d0 - rho)*dafm(i,j) + rho*dafp(i,j)
                    endif

                 endif
              endif

           else
              ! if Eqs. 24 or 25 didn't hold we still may need to limit
              if (abs(dafm(i,j)) >= 2.0d0*abs(dafp(i,j))) then
                 ar(i,j) = a(i,j) - 2.0d0*dafp(i,j)
              endif
              if (abs(dafp(i,j)) >= 2.0d0*abs(dafm(i,j))) then
                 al(i+1,j) = a(i,j) + 2.0d0*dafm(i,j)
              endif
           endif

        enddo
     enddo

  else if (idir == 2) then

     do j = jlo-2, jhi+3
        do i = ilo, ihi

           ! interpolate to the edges
           a_int(i,j) = (7.0/12.0)*(a(i,j-1) + a(i,j)) - (1.0/12.0)*(a(i,j-2) + a(i,j+1))

           al(i,j) = a_int(i,j)
           ar(i,j) = a_int(i,j)

        enddo
     enddo

     do j = jlo-1, jhi+2
        do i = ilo, ihi
           ! these live on cell-centers
           dafm(i,j) = a(i,j) - a_int(i,j)
           dafp(i,j) = a_int(i,j+1) - a(i,j)

           ! these live on cell-centers
           d2af(i,j) = 6.0*(a_int(i,j) - 2.0*a(i,j) + a_int(i,j+1))
           d2ac(i,j) = a(i,j-1) - 2.0*a(i,j) + a(i,j+1)

           ! this lives on the interface
           d3a(i,j) = d2ac(i,j) - d2ac(i,j-1)

        enddo
     enddo

     do j = jlo, jhi+1
        do i = ilo, ihi

           ! limit? MC Eq. 24 and 25
           if (dafm(i,j) * dafp(i,j) <= 0.0 .or. &
                (a(i,j) - a(i,j-2))*(a(i,j+2) - a(i,j)) <= 0.0) then

              ! we are at an extrema

              s = sign(1.0, d2ac(i,j))
              if (s == sign(1.0, d2ac(i,j-1)) .and. s == sign(1.0, d2ac(i,j+1)) .and. &
                  s == sign(1.0, d2af(i,j))) then
                 ! MC Eq. 26
                 d2a_lim = s*min(abs(d2af(i,j)), C2*abs(d2ac(i,j-1)), &
                                 C2*abs(d2ac(i,j)), C2*abs(d2ac(i,j+1)))
              else
                 d2a_lim = 0.0d0
              endif

              if (abs(d2af(i,j)) <= 1.e-12*max(abs(a(i,j-2)), abs(a(i,j-1)), &
                                               abs(a(i,j)), abs(a(i,j+1)), abs(a(i,j+2)))) then
                 rho = 0.0
              else
                 ! MC Eq. 27
                 rho = d2a_lim/d2af(i,j)
              enddo

              if (rho < 1.0d0 - 1.d-12) then
                 ! we may need to limit -- these quantities are at cell-centers
                 d3a_min = min(d3a(i,j-1), d3a(i,j), d3a(i,j+1), d3a(i,j+2))
                 d3a_max = max(d3a(i,j-1), d3a(i,j), d3a(i,j+1), d3a(i,j+2))

                 if (C3*max(abs(d3a_min), abs(d3a_max)) <= (d3a_max - d3a_min)) then
                    ! limit
                    if (dafm(i,j)*dafp(i,j) < 0.0d0) then
                       ! Eqs. 29, 30
                       ar(i,j) = a(i,j) - rho*dafm(i,j)  ! note: typo in Eq 29
                       al(i,j+1) = a(i,j) + rho*dafp(i,j)
                    else if (abs(dafm(i,j)) >= 2.0*abs(dafp(i,j))) then
                       ! Eq. 31
                       ar(i,j) = a(i,j) - 2.0d0*(1.0d0 - rho)*dafp(i,j) - rho*dafm(i,j)
                    else if (abs(dafp(i,j)) >= 2.0*abs(dafm(i,j))) then
                       ! Eq. 32
                       al(i,j+1) = a(i,j) + 2.0d0*(1.0d0 - rho)*dafm(i,j) + rho*dafp(i,j)
                    endif

                 endif
              endif

           else
              ! if Eqs. 24 or 25 didn't hold we still may need to limit
              if (abs(dafm(i,j)) >= 2.0d0*abs(dafp(i,j))) then
                 ar(i,j) = a(i,j) - 2.0d0*dafp(i,j)
              endif
              if (abs(dafp(i,j)) >= 2.0d0*abs(dafm(i,j))) then
                 al(i,j+1) = a(i,j) + 2.0d0*dafm(i,j)
              endif
           endif

        enddo
     enddo

  endif
