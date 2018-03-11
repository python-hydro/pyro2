subroutine mac_vels(qx, qy, ng, dx, dy, dt, &
                    u, v, &
                    ldelta_ux, ldelta_vx, &
                    ldelta_uy, ldelta_vy, &
                    gradp_x, gradp_y, &
                    u_MAC, v_MAC)

  implicit none

  integer, intent(in) :: qx, qy, ng
  double precision, intent(in) :: dx, dy, dt

  ! 0-based indexing to match python
  double precision, intent(inout) :: u(0:qx-1, 0:qy-1)
  double precision, intent(inout) :: v(0:qx-1, 0:qy-1)

  double precision, intent(inout) :: ldelta_ux(0:qx-1, 0:qy-1)
  double precision, intent(inout) :: ldelta_vx(0:qx-1, 0:qy-1)

  double precision, intent(inout) :: ldelta_uy(0:qx-1, 0:qy-1)
  double precision, intent(inout) :: ldelta_vy(0:qx-1, 0:qy-1)

  double precision, intent(inout) :: gradp_x(0:qx-1, 0:qy-1)
  double precision, intent(inout) :: gradp_y(0:qx-1, 0:qy-1)

  double precision, intent(  out) :: u_MAC(0:qx-1, 0:qy-1)
  double precision, intent(  out) :: v_MAC(0:qx-1, 0:qy-1)

!f2py depend(qx, qy) :: u, v
!f2py depend(qx, qy) :: ldelta_ux, ldelta_vx, ldelta_uy, ldelta_vy
!f2py depend(qx, qy) :: gradp_x, gradp_y
!f2py depend(qx, qy) :: utrans, vtrans
!f2py depend(qx, qy) :: u_MAC, v_MAC
!f2py intent(in) :: u, v, gradp_x, gradp_y
!f2py intent(in) :: ldelta_ux, ldelta_vx, ldelta_uy, ldelta_vy
!f2py intent(out) :: u_MAC, v_MAC


  double precision :: u_xl(0:qx-1, 0:qy-1), u_xr(0:qx-1, 0:qy-1)
  double precision :: u_yl(0:qx-1, 0:qy-1), u_yr(0:qx-1, 0:qy-1)
  double precision :: v_xl(0:qx-1, 0:qy-1), v_xr(0:qx-1, 0:qy-1)
  double precision :: v_yl(0:qx-1, 0:qy-1), v_yr(0:qx-1, 0:qy-1)

  
  ! get the full u and v left and right states (including transverse
  ! terms) on both the x- and y-interfaces
  call get_interface_states(qx, qy, ng, dx, dy, dt, &
                            u, v, &
                            ldelta_ux, ldelta_vx, &
                            ldelta_uy, ldelta_vy, &
                            gradp_x, gradp_y, &
                            u_xl, u_xr, u_yl, u_yr, &
                            v_xl, v_xr, v_yl, v_yr)
  

  ! Riemann problem -- this follows Burger's equation.  We don't use
  ! any input velocity for the upwinding.  Also, we only care about
  ! the normal states here (u on x and v on y)
  call riemann_and_upwind(qx, qy, ng, u_xl, u_xr, u_MAC)
  call riemann_and_upwind(qx, qy, ng, v_yl, v_yr, v_MAC)
    
end subroutine mac_vels


!xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
subroutine states(qx, qy, ng, dx, dy, dt, &
                  u, v, &
                  ldelta_ux, ldelta_vx, &
                  ldelta_uy, ldelta_vy, &
                  gradp_x, gradp_y, &
                  u_MAC, v_MAC, &
                  u_xint, v_xint, u_yint, v_yint)

  ! this is similar to mac_vels, but it predicts the interface states
  ! of both u and v on both interfaces, using the MAC velocities to
  ! do the upwinding.

  implicit none

  integer, intent(in) :: qx, qy, ng
  double precision, intent(in) :: dx, dy, dt

  ! 0-based indexing to match python
  double precision, intent(inout) :: u(0:qx-1, 0:qy-1)
  double precision, intent(inout) :: v(0:qx-1, 0:qy-1)

  double precision, intent(inout) :: ldelta_ux(0:qx-1, 0:qy-1)
  double precision, intent(inout) :: ldelta_vx(0:qx-1, 0:qy-1)

  double precision, intent(inout) :: ldelta_uy(0:qx-1, 0:qy-1)
  double precision, intent(inout) :: ldelta_vy(0:qx-1, 0:qy-1)

  double precision, intent(inout) :: gradp_x(0:qx-1, 0:qy-1)
  double precision, intent(inout) :: gradp_y(0:qx-1, 0:qy-1)

  double precision, intent(inout) :: u_MAC(0:qx-1, 0:qy-1)
  double precision, intent(inout) :: v_MAC(0:qx-1, 0:qy-1)

  double precision, intent(out) :: u_xint(0:qx-1, 0:qy-1), u_yint(0:qx-1, 0:qy-1)  
  double precision, intent(out) :: v_xint(0:qx-1, 0:qy-1), v_yint(0:qx-1, 0:qy-1)  

!f2py depend(qx, qy) :: u, v
!f2py depend(qx, qy) :: ldelta_ux, ldelta_vx, ldelta_uy, ldelta_vy
!f2py depend(qx, qy) :: gradp_x, gradp_y
!f2py depend(qx, qy) :: utrans, vtrans
!f2py depend(qx, qy) :: u_MAC, v_MAC
!f2py intent(in) :: u, v, gradp_x, gradp_y
!f2py intent(in) :: ldelta_ux, ldelta_vx, ldelta_uy, ldelta_vy
!f2py intent(in) :: u_MAC, v_MAC
!f2py intent(out) :: u_xint, v_xint, u_yint, v_yint

  double precision :: u_xl(0:qx-1, 0:qy-1), u_xr(0:qx-1, 0:qy-1)
  double precision :: u_yl(0:qx-1, 0:qy-1), u_yr(0:qx-1, 0:qy-1)
  double precision :: v_xl(0:qx-1, 0:qy-1), v_xr(0:qx-1, 0:qy-1)
  double precision :: v_yl(0:qx-1, 0:qy-1), v_yr(0:qx-1, 0:qy-1)


  ! get the full u and v left and right states (including transverse
  ! terms) on both the x- and y-interfaces
  call get_interface_states(qx, qy, ng, dx, dy, dt, &
                            u, v, &
                            ldelta_ux, ldelta_vx, &
                            ldelta_uy, ldelta_vy, &
                            gradp_x, gradp_y, &
                            u_xl, u_xr, u_yl, u_yr, &
                            v_xl, v_xr, v_yl, v_yr)


  ! upwind using the MAC velocity to determine which state exists on
  ! the interface
  call upwind(qx, qy, ng, u_xl, u_xr, u_MAC, u_xint)
  call upwind(qx, qy, ng, v_xl, v_xr, u_MAC, v_xint)
  call upwind(qx, qy, ng, u_yl, u_yr, v_MAC, u_yint)
  call upwind(qx, qy, ng, v_yl, v_yr, v_MAC, v_yint)
    
end subroutine states


!xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
subroutine get_interface_states(qx, qy, ng, dx, dy, dt, &
                                u, v, &
                                ldelta_ux, ldelta_vx, &
                                ldelta_uy, ldelta_vy, &
                                gradp_x, gradp_y, &
                                u_xl, u_xr, u_yl, u_yr, &
                                v_xl, v_xr, v_yl, v_yr)

  ! Compute the unsplit predictions of u and v on both the x- and
  ! y-interfaces.  This includes the transverse terms.

  implicit none

  integer, intent(in) :: qx, qy, ng
  double precision, intent(in) :: dx, dy, dt

  ! 0-based indexing to match python
  double precision, intent(inout) :: u(0:qx-1, 0:qy-1)
  double precision, intent(inout) :: v(0:qx-1, 0:qy-1)

  double precision, intent(inout) :: ldelta_ux(0:qx-1, 0:qy-1)
  double precision, intent(inout) :: ldelta_vx(0:qx-1, 0:qy-1)

  double precision, intent(inout) :: ldelta_uy(0:qx-1, 0:qy-1)
  double precision, intent(inout) :: ldelta_vy(0:qx-1, 0:qy-1)

  double precision, intent(inout) :: gradp_x(0:qx-1, 0:qy-1)
  double precision, intent(inout) :: gradp_y(0:qx-1, 0:qy-1)

  double precision, intent(out) :: u_xl(0:qx-1, 0:qy-1), u_xr(0:qx-1, 0:qy-1)
  double precision, intent(out) :: u_yl(0:qx-1, 0:qy-1), u_yr(0:qx-1, 0:qy-1)
  double precision, intent(out) :: v_xl(0:qx-1, 0:qy-1), v_xr(0:qx-1, 0:qy-1)
  double precision, intent(out) :: v_yl(0:qx-1, 0:qy-1), v_yr(0:qx-1, 0:qy-1)

  integer :: ilo, ihi, jlo, jhi
  integer :: nx, ny
  integer :: i, j

  double precision :: uhat_adv(0:qx-1, 0:qy-1), vhat_adv(0:qx-1, 0:qy-1)

  double precision :: u_xint(0:qx-1, 0:qy-1), u_yint(0:qx-1, 0:qy-1)  
  double precision :: v_xint(0:qx-1, 0:qy-1), v_yint(0:qx-1, 0:qy-1)  

  double precision :: dtdx, dtdy
  double precision :: ubar, vbar, uv_x, vu_y, uu_x, vv_y

  nx = qx - 2*ng; ny = qy - 2*ng
  ilo = ng; ihi = ng+nx-1; jlo = ng; jhi = ng+ny-1


  ! first predict u and v to both interfaces, considering only the normal
  ! part of the predictor.  These are the 'hat' states. 
  

  dtdx = dt/dx
  dtdy = dt/dy

  do j = jlo-2, jhi+2
     do i = ilo-2, ihi+2        

        ! u on x-edges
        u_xl(i+1,j) = u(i,j) + 0.5d0*(1.0d0 - dtdx*u(i,j))*ldelta_ux(i,j)
        u_xr(i  ,j) = u(i,j) - 0.5d0*(1.0d0 + dtdx*u(i,j))*ldelta_ux(i,j)

        ! v on x-edges 
        v_xl(i+1,j) = v(i,j) + 0.5d0*(1.0d0 - dtdx*u(i,j))*ldelta_vx(i,j)
        v_xr(i  ,j) = v(i,j) - 0.5d0*(1.0d0 + dtdx*u(i,j))*ldelta_vx(i,j)

        ! u on y-edges 
        u_yl(i,j+1) = u(i,j) + 0.5d0*(1.0d0 - dtdy*v(i,j))*ldelta_uy(i,j)
        u_yr(i,j  ) = u(i,j) - 0.5d0*(1.0d0 + dtdy*v(i,j))*ldelta_uy(i,j)

        ! v on y-edges
        v_yl(i,j+1) = v(i,j) + 0.5d0*(1.0d0 - dtdy*v(i,j))*ldelta_vy(i,j)
        v_yr(i,j  ) = v(i,j) - 0.5d0*(1.0d0 + dtdy*v(i,j))*ldelta_vy(i,j)

     enddo
  enddo


  ! now get the normal advective velocities on the interfaces by solving
  ! the Riemann problem.  
  call riemann(qx, qy, ng, u_xl, u_xr, uhat_adv)
  call riemann(qx, qy, ng, v_yl, v_yr, vhat_adv)


  ! now that we have the advective velocities, upwind the left and right
  ! states using the appropriate advective velocity.

  ! on the x-interfaces, we upwind based on uhat_adv
  call upwind(qx, qy, ng, u_xl, u_xr, uhat_adv, u_xint)
  call upwind(qx, qy, ng, v_xl, v_xr, uhat_adv, v_xint)

  ! on the y-interfaces, we upwind based on vhat_adv
  call upwind(qx, qy, ng, u_yl, u_yr, vhat_adv, u_yint)
  call upwind(qx, qy, ng, v_yl, v_yr, vhat_adv, v_yint)

  ! at this point, these states are the `hat' states -- they only
  ! considered the normal to the interface portion of the predictor.


  ! add the transverse flux differences to the preliminary interface states
  do j = jlo-2, jhi+2
     do i = ilo-2, ihi+2        

        ubar = 0.5d0*(uhat_adv(i,j) + uhat_adv(i+1,j))
        vbar = 0.5d0*(vhat_adv(i,j) + vhat_adv(i,j+1))

        ! v du/dy is the transerse term for the u states on x-interfaces
        vu_y = vbar*(u_yint(i,j+1) - u_yint(i,j))
        
        u_xl(i+1,j) = u_xl(i+1,j) - 0.5*dtdy*vu_y - 0.5*dt*gradp_x(i,j)
        u_xr(i  ,j) = u_xr(i  ,j) - 0.5*dtdy*vu_y - 0.5*dt*gradp_x(i,j)

        ! v dv/dy is the transverse term for the v states on x-interfaces
        vv_y = vbar*(v_yint(i,j+1) - v_yint(i,j))

        v_xl(i+1,j) = v_xl(i+1,j) - 0.5*dtdy*vv_y - 0.5*dt*gradp_y(i,j)
        v_xr(i  ,j) = v_xr(i  ,j) - 0.5*dtdy*vv_y - 0.5*dt*gradp_y(i,j)

        ! u dv/dx is the transverse term for the v states on y-interfaces
        uv_x = ubar*(v_xint(i+1,j) - v_xint(i,j))

        v_yl(i,j+1) = v_yl(i,j+1) - 0.5*dtdx*uv_x - 0.5*dt*gradp_y(i,j)
        v_yr(i,j  ) = v_yr(i,j  ) - 0.5*dtdx*uv_x - 0.5*dt*gradp_y(i,j)

        ! u du/dx is the transverse term for the u states on y-interfaces
        uu_x = ubar*(u_xint(i+1,j) - u_xint(i,j))

        u_yl(i,j+1) = u_yl(i,j+1) - 0.5*dtdx*uu_x - 0.5*dt*gradp_x(i,j)
        u_yr(i,j  ) = u_yr(i,j  ) - 0.5*dtdx*uu_x - 0.5*dt*gradp_x(i,j)

     enddo
  enddo

end subroutine get_interface_states


!xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
subroutine upwind(qx, qy, ng, q_l, q_r, s, q_int)

  ! upwind the left and right states based on the specified input
  ! velocity, s.  The resulting interface state is q_int

  implicit none

  integer :: qx, qy, ng
  double precision :: q_l(0:qx-1, 0:qy-1), q_r(0:qx-1, 0:qy-1)
  double precision :: s(0:qx-1, 0:qy-1)
  double precision :: q_int(0:qx-1, 0:qy-1)

  integer :: ilo, ihi, jlo, jhi
  integer :: nx, ny
  integer :: i, j

  nx = qx - 2*ng; ny = qy - 2*ng
  ilo = ng; ihi = ng+nx-1; jlo = ng; jhi = ng+ny-1

  do j = jlo-1, jhi+1
     do i = ilo-1, ihi+1

        if (s(i,j) > 0.0d0) then
           q_int(i,j) = q_l(i,j)
        else if (s(i,j) == 0.0d0) then
           q_int(i,j) = 0.5d0*(q_l(i,j) + q_r(i,j))
        else
           q_int(i,j) = q_r(i,j)
        endif

     enddo
  enddo

end subroutine upwind


!xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
subroutine riemann(qx, qy, ng, q_l, q_r, s)

  ! Solve the Burger's Riemann problem given the input left and right
  ! states and return the state on the interface.
  !
  ! This uses the expressions from Almgren, Bell, and Szymczak 1996.

  implicit none

  integer :: qx, qy, ng
  double precision :: q_l(0:qx-1, 0:qy-1), q_r(0:qx-1, 0:qy-1)
  double precision :: s(0:qx-1, 0:qy-1)

  integer :: ilo, ihi, jlo, jhi
  integer :: nx, ny
  integer :: i, j

  nx = qx - 2*ng; ny = qy - 2*ng
  ilo = ng; ihi = ng+nx-1; jlo = ng; jhi = ng+ny-1

  do j = jlo-1, jhi+1
     do i = ilo-1, ihi+1

        if (q_l(i,j) > 0.0d0 .and. q_l(i,j) + q_r(i,j) > 0.0d0) then
           s(i,j) = q_l(i,j)
        else if (q_l(i,j) <= 0.0d0 .and. q_r(i,j) >= 0.0d0) then
           s(i,j) = 0.0d0
        else
           s(i,j) = q_r(i,j)
        endif

     enddo
  enddo

end subroutine riemann


!xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
subroutine riemann_and_upwind(qx, qy, ng, q_l, q_r, q_int)

  ! First solve the Riemann problem given q_l and q_r to give the 
  ! velocity on the interface and then use this velocity to upwind to
  ! determine the state (q_l, q_r, or a mix) on the interface).
  !
  ! This differs from upwind, above, in that we don't take in a
  ! velocity to upwind with).

  implicit none

  integer :: qx, qy, ng
  double precision :: q_l(0:qx-1, 0:qy-1), q_r(0:qx-1, 0:qy-1)
  double precision :: q_int(0:qx-1, 0:qy-1)

  double precision :: s(0:qx-1, 0:qy-1)

  call riemann(qx, qy, ng, q_l, q_r, s)
  call upwind(qx, qy, ng, q_l, q_r, s, q_int)

end subroutine riemann_and_upwind
