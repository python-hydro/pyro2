function f(p, U_ij, gamma, idens, ixmom, iymom, iener, nvar) result (root)
    ! function for doing root finding

    implicit none

    double precision :: root
    double precision :: p, gamma, U_ij(0:nvar-1)
    integer :: idens, ixmom, iymom, iener, nvar


    double precision :: D, tau, u, v, W

    D = U_ij(idens)
    tau = U_ij(iener)

    if (abs(tau+p) < 1.0d-6) then
        u = U_ij(ixmom)
        v = U_ij(iymom)
    else
        u = U_ij(ixmom) / (tau + p + D)
        v = U_ij(iymom) / (tau + p + D)
    endif

    W = 1.0d0 / sqrt(1.0d0 - u**2 - v**2)

    root = (gamma-1.0d0) * (tau + D*(1.0d0-W) + p*(1.0d0-W**2)) / W**2 - p

end function f

function brentq(x1, b, U, gamma, idens, ixmom, iymom, iener, nvar) result (p)
    ! route finder using brent's method
    implicit none

    double precision :: p
    double precision  :: U(0:nvar-1), x1, gamma
    double precision :: b
    integer :: idens, ixmom, iymom, iener, nvar

    double precision, parameter :: TOL = 1.0d-6
    integer, parameter :: ITMAX = 100

    double precision :: a, c, d, fa, fb, fc, fs, s
    logical :: mflag, con1, con2, con3, con4, con5
    integer :: i

    double precision :: f

    a = x1
    c = 0.0d0
    d = 0.0d0
    fa = f(a, U, gamma, idens, ixmom, iymom, iener, nvar)
    fb = f(b, U, gamma, idens, ixmom, iymom, iener, nvar)
    fc = 0.0d0

    if (fa * fb >= 0.0d0) then
        p = x1
        return
    end if

    if (abs(fa) < abs(fb)) then
        d = a
        a = b
        b = d

        d = fa
        fa = fb
        fb = d
    end if

    c = a
    fc = fa

    mflag = .true.

    do i = 1, ITMAX
        if (fa /= fc .and. fb /= fc) then
            s = a*fb*fc / ((fa-fb) * (fa-fc)) + b*fa*fc / ((fb-fa)*(fb-fc)) +&
                c*fa*fb / ((fc-fa)*(fc-fb))
        else
            s = b - fb * (b-a) / (fb-fa)
        end if

        con1 = .false.

        if (0.25d0 * (3.0d0 * a + b) < b) then
            if ( s < 0.25d0 * (3.0d0 * a + b) .or. s > b) then
                con1 = .true.
            end if
        else if (s < b .or. s > 0.25d0  * (3.0d0 * a + b)) then
            con1 = .true.
        end if

        con2 = mflag .and. abs(s - b) >= 0.5d0 * abs(b-c)

        con3 = (.not. mflag) .and. abs(s-b) >= 0.5d0 * abs(c-d)

        con4 = mflag .and. abs(b-c) < TOL

        con5 = (.not. mflag) .and. abs(c-d) < TOL

        if (con1 .or. con2 .or. con3 .or. con4 .or. con5) then
            s = 0.5d0 * (a + b)
            mflag = .true.
        else
            mflag = .false.
        end if

        fs = f(s, U, gamma, idens, ixmom, iymom, iener, nvar)

        if (abs(fa) < abs(fb)) then
            d = a
            a = b
            b = d

            d = fa
            fa = fb
            fb = d
        end if

        d = c
        c = b
        fc = fb

        if (fa * fs < 0.0d0) then
            b = s
            fb = fs
        else
            a = s
            fa = fs
        end if

        if (fb == 0.0d0 .or. fs == 0.0d0 .or. abs(b-a) < TOL) then
            p = b
            return
        end if

    end do

    p = x1

end function brentq

subroutine cons_to_prim(U, qx, qy, &
                        irho, iu, iv, ip, ix, irhox, &
                        idens, ixmom, iymom, iener, nvar, &
                        naux, gamma, q)
    ! convert an input vector of conserved variables to primitive variables

    implicit none

    integer, intent(in) :: qx, qy
    integer, intent(in) :: irho, iu, iv, ip, ix, irhox
    integer, intent(in) :: idens, ixmom, iymom, iener, nvar, naux
    double precision, intent(in) :: gamma
    double precision, intent(in) :: U(0:qx-1, 0:qy-1, 0:nvar-1)
    double precision, intent(out) :: q(0:qx-1, 0:qy-1, 0:nvar-1)

!f2py depend(qx, qy, nvar) :: q, U
!f2py intent(in) :: U
!f2py intent(out) :: q

    integer :: i, j
    double precision :: pmin, pmax, fmin, fmax
    double precision :: W(0:qx-1, 0:qy-1)

    double precision :: f, brentq

    double precision, parameter :: smallp = 1.0d-6

    do j = 0, qx-1
        do i = 0, qy-1
            pmax = max((gamma-1.0d0)*U(i, j, iener)*1.0000000001d0, smallp)

            pmin = max(min(1.0d-6*pmax, smallp), sqrt(U(i, j, ixmom)**2+U(i, j, iymom)**2) - U(i, j, iener) - U(i, j, idens))

            fmin = f(pmin, U(i, j, :), gamma, idens, ixmom, iymom, iener, nvar)
            fmax = f(pmax, U(i, j, :), gamma, idens, ixmom, iymom, iener, nvar)

            if (fmin * fmax > 0.0d0) then
                pmin = pmin * 1.0d-2
                fmin = f(pmin, U(i, j, :), gamma, idens, ixmom, iymom, iener, nvar)
            endif

            if (fmin * fmax > 0.0d0) then
                pmax = min(pmax*1.0d2, 1.0d0)
            endif

            if (fmin * fmax > 0.0d0) then
                q(i, j, ip) = max((gamma-1.0d0)*U(i, j, iener), smallp)
            else
                ! try:
                q(i, j, ip) = brentq(pmin, pmax, U(i,j,:), gamma, idens, ixmom, iymom, iener, nvar)
            endif

            if ((q(i, j, ip) /= q(i, j, ip)) .or. &
                (q(i, j, ip)-1 == q(i, j, ip)) .or. &
                (abs(q(i, j, ip)) > 1.0e10)) then ! nan or infty alert
                q(i, j, ip) = max((gamma-1.0d0)*U(i, j, iener), smallp)
            endif

            ! except ValueError:
            !     q(i, j, ip) = max((gamma-1.0d0)*U(i, j, iener), 0.0d0)

            if (abs(U(i, j, iener) + U(i, j, idens) + q(i, j, ip)) < 1.0d-5) then
                q(i, j, iu) = U(i, j, ixmom)
                q(i, j, iv) = U(i, j, iymom)
            else
                q(i, j, iu) = U(i, j, ixmom)/(U(i, j, iener) + U(i, j, idens) + q(i, j, ip))
                q(i, j, iv) = U(i, j, iymom)/(U(i, j, iener) + U(i, j, idens) + q(i, j, ip))
            endif

            ! nan check
            if (q(i,j,iu) /= q(i,j,iu)) then
                q(i,j,iu) = 0.0d0
            endif
            if (q(i,j,iv) /= q(i,j,iv)) then
                q(i,j,iv) = 0.0d0
            endif

        enddo
    enddo

    W = 1.0d0/sqrt(1.0d0 - q(:, :, iu)**2 - q(:, :, iv)**2)

    q(:, :, irho) = U(:, :, idens) / W

    if (naux > 0) then
        do i = 0, naux-1
            q(:, :, ix+i) = U(:, :, irhox+i)/(q(:, :, irho) * W)
        enddo
    endif

end subroutine cons_to_prim
