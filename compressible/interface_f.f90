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

