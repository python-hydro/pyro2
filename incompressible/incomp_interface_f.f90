subroutine trans_vels(qx, qy, ng, dx, dy, dt, &
                      u, v, &
                      ldelta_u, ldelta_v, &
                      utrans, vtrans)

  implicit none

  integer, intent(in) :: qx, qy, ng
  double precision, intent(in) :: dx, dy, dt

  ! 0-based indexing to match python
  double precision, intent(inout) :: u(0:qx-1, 0:qy-1)
  double precision, intent(inout) :: v(0:qx-1, 0:qy-1)

  double precision, intent(inout) :: ldelta_u(0:qx-1, 0:qy-1)
  double precision, intent(inout) :: ldelta_v(0:qx-1, 0:qy-1)

  double precision, intent(  out) :: utrans(0:qx-1, 0:qy-1)
  double precision, intent(  out) :: vtrans(0:qx-1, 0:qy-1)

!f2py depend(qx, qy) :: u, v, ldelta_u, ldelta_v
!f2py depend(qx, qy) :: utrans, vtrans
!f2py intent(in) :: u, v, ldelta_u, ldelta_v
!f2py intent(out) :: utrans, vtrans
 
  ! construct the transverse states of u and v --- these will be
  ! used in constructing the transverse flux difference for the 
  ! full interface states

  ! specifically, we need u on the x-interfaces (for the transverse
  ! term in the v normal state) and v on the y-interfaces (for the
  ! transverse term in the u normal state)

  integer :: ilo, ihi, jlo, jhi
  integer :: nx, ny
  integer :: i, j

  double precision :: q_l(0:qx-1, 0:qy-1), q_r(0:qx-1, 0:qy-1)

  

  nx = qx - 2*ng; ny = qy - 2*ng
  ilo = ng; ihi = ng+nx-1; jlo = ng; jhi = ng+ny-1


  ! x-interface states of u
  dtdx = dt/dx

  do j = jlo-2, jhi+2
     do i = ilo-2, ihi+2
        
        q_l(i+1,j) = u(i,j) + 0.5d0*(1.0d0 - dtdx*u(i,j))*ldelta_u(i,j)
        q_r(i  ,j) = u(i,j) - 0.5d0*(1.0d0 + dtdx*u(i,j))*ldelta_u(i,j)

     enddo
  enddo

  ! Riemann problem -- this is based on Burger's equation.  See 
  ! Bell, Colella, and Howell (1991), Eq. 3.3, or Almgren, Bell,
  ! and Szymczak (1996) (top of page 362)
  do j = jlo-1, jhi+1
     do i = ilo-1, ihi+1
        if (q_l(i,j) > 0.0d0 .and. (q_l(i,j) + q_r(i,j)) > 0.0d0) then
           utrans(i,j) = q_l(i,j)
        else if (q_l(i,j) <= 0.0d0 .and. q_r(i,j) >= 0.0d0) then
           utrans(i,j) = 0.0d0
        else
           utrans(i,j) = q_r(i,j)
        endif
     enddo
  enddo


  ! y-interface states of v
  dtdy = dt/dy

  do j = jlo-2, jhi+2
     do i = ilo-2, ihi+2
        
        q_l(i,j+1) = v(i,j) + 0.5d0*(1.0d0 - dtdy*v(i,j))*ldelta_v(i,j)
        q_r(i,j  ) = v(i,j) - 0.5d0*(1.0d0 + dtdy*v(i,j))*ldelta_v(i,j)

     enddo
  enddo

  ! Riemann problem 
  do j = jlo-1, jhi+1
     do i = ilo-1, ihi+1
        if (q_l(i,j) > 0.0d0 .and. (q_l(i,j) + q_r(i,j)) > 0.0d0) then
           vtrans(i,j) = q_l(i,j)
        else if (q_l(i,j) <= 0.0d0 .and. q_r(i,j) >= 0.0d0) then
           vtrans(i,j) = 0.0d0
        else
           vtrans(i,j) = q_r(i,j)
        endif
     enddo
  enddo

end subroutine trans_vels



subroutine mac_vels(qx, qy, ng, dx, dy, dt, &
                    u, v, &
                    ldelta_u, ldelta_v, &
                    gradp_x, gradp_y,
                    utrans, vtrans,
                    u_MAC, v_MAC)

  implicit none

  integer, intent(in) :: qx, qy, ng
  double precision, intent(in) :: dx, dy, dt

  ! 0-based indexing to match python
  double precision, intent(inout) :: u(0:qx-1, 0:qy-1)
  double precision, intent(inout) :: v(0:qx-1, 0:qy-1)

  double precision, intent(inout) :: ldelta_u(0:qx-1, 0:qy-1)
  double precision, intent(inout) :: ldelta_v(0:qx-1, 0:qy-1)

  double precision, intent(inout) :: gradp_x(0:qx-1, 0:qy-1)
  double precision, intent(inout) :: gradp_y(0:qx-1, 0:qy-1)

  double precision, intent(inout) :: utrans(0:qx-1, 0:qy-1)
  double precision, intent(inout) :: vtrans(0:qx-1, 0:qy-1)

  double precision, intent(  out) :: u_MAC(0:qx-1, 0:qy-1)
  double precision, intent(  out) :: v_MAC(0:qx-1, 0:qy-1)

!f2py depend(qx, qy) :: u, v, ldelta_u, ldelta_v
!f2py depend(qx, qy) :: gradp_x, gradp_y
!f2py depend(qx, qy) :: utrans, vtrans
!f2py depend(qx, qy) :: u_MAC, v_MAC
!f2py intent(in) :: u, v, ldelta_u, ldelta_v, gradp_x, gradp_y, utrans, vtrans
!f2py intent(out) :: u_MAC, v_MAC

  integer :: ilo, ihi, jlo, jhi
  integer :: nx, ny
  integer :: i, j

  double precision :: q_l(0:qx-1, 0:qy-1), q_r(0:qx-1, 0:qy-1)

  nx = qx - 2*ng; ny = qy - 2*ng
  ilo = ng; ihi = ng+nx-1; jlo = ng; jhi = ng+ny-1
  
  ! u on x-edges
  do j = jlo-2, jhi+2
     do i = ilo-2, ihi+2        
        q_l(i+1,j) = u(i,j) + 0.5d0*(1.0d0 - dtdx*u(i,j))*ldelta_u(i,j)
        q_r(i  ,j) = u(i,j) - 0.5d0*(1.0d0 + dtdx*u(i,j))*ldelta_u(i,j)
     enddo
  enddo

  ! Riemann problem -- use utrans for the upwinding velocity
  call riemann_upwind()


  ! v on y-edges


  ! Riemann problem -- use vtrans for the upwinding velocity
  call riemann_upwind()


  ! transverse flux differences 


  ! Riemann problem -- this follows Burger's equation
