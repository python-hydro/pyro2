import numpy as np
from numba import njit


@njit(cache=True)
def f(p, U_ij, gamma, idens, ixmom, iymom, iener):
    """
    Function whose root needs to be found for cons to prim 
    """

    D = U_ij[idens]
    tau = U_ij[iener]

    if abs(tau+p) < 1.e-6:
        u = U_ij[ixmom]
        v = U_ij[iymom]
    else:
        u = U_ij[ixmom] / (tau + p + D)
        v = U_ij[iymom] / (tau + p + D)

    # Lorentz factor 
    W = 1.0 / np.sqrt(1.0 - u**2 - v**2)

    return (gamma - 1.0) * (tau + D*(1.0-W) + p*(1.0-W**2)) / W**2 - p


@njit(cache=True)
def brentq(x1, b, U, gamma, idens, ixmom, iymom, iener, 
           TOL=1.e-6, ITMAX=100):
    """
    Root finder using Brent's method
    """

    # initialize variables
    a = x1
    c = 0.0
    d = 0.0
    fa = f(a, U, gamma, idens, ixmom, iymom, iener)
    fb = f(b, U, gamma, idens, ixmom, iymom, iener)
    fc = 0.0

    # root found
    if fa * fb >= 0.0:
        return x1 

    # switch variables 
    if abs(fa) < abs(fb):
        d = a
        a = b 
        b = d 

        d = fa 
        fa = fb
        fb = d 

    c = a 
    fc = fa 

    mflag = True

    for i in range(ITMAX):
        if fa != fc and fb != fc:
            s = a*fb*fc / ((fa-fb) * (fa-fc)) + b*fa*fc / ((fb-fa)*(fb-fc)) + \
                c*fa*fb / ((fc-fa)*(fc-fb))
        else:
            s = b - fb * (b-a) / (fb-fa)

        # test conditions and store in con1-con5
        con1 = False 

        if 0.25 * (3.0 * a + b) < b:
            if s < 0.25 * (3.0 * a + b) or s > b:
                con1 = True 
        elif s < b or s > 0.25 * (3.0 * a + b):
            con1 = True 

        con2 = mflag and abs(s-b) >= 0.5 * abs(b-c)

        con3 = (not mflag) and abs(s-b) >= 0.5 * abs(c-d)

        con4 = mflag and abs(b-c) < TOL 

        con5 = (not mflag) and abs(c-d) < TOL 

        if con1 or con2 or con3 or con4 or con5:
            s = 0.5 * (a + b)
            mflag = True 
        else:
            mflag = False 

        # evaluate at midpoint and set new limits 
        fs = f(s, U, gamma, idens, ixmom, iymom, iener)

        if abs(fa) < abs(fb):
            d = a
            a = b 
            b = d 

            d = fa 
            fa = fb 
            fb = d 

        d = c 
        c = b 
        fc = fb

        if fa * fs < 0.0:
            b = s 
            fb = fs 
        else:
            a = s 
            fa = fs 

        # found solution to required tolerance
        if fb == 0.0 or fs == 0.0 or abs(b-a) < TOL:
            return b 

    return x1 


@njit(cache=True)
def cons_to_prim(U, 
                 irho, iu, iv, ip, ix, irhox, 
                 idens, ixmom, iymom, iener, 
                 naux, gamma, q, smallp=1.e-6):
    """
    convert an input vector of conserved variables to primitive variables 
    """

    qx, qy, _ = U.shape

    for j in range(qy):
        for i in range(qx):
            pmax = max((gamma-1.0)*U[i, j, iener]*1.0000000001, smallp)

            pmin = max(min(1.0e-6*pmax, smallp), np.sqrt(U[i, j, ixmom] **
                       2+U[i, j, iymom]**2) - U[i, j, iener] - U[i, j, idens])

            fmin = f(pmin, U[i, j, :], gamma, idens, ixmom, iymom, iener)
            fmax = f(pmax, U[i, j, :], gamma, idens, ixmom, iymom, iener)

            if fmin * fmax > 0.0:
                pmin = pmin * 1.0e-2
                fmin = f(pmin, U[i, j, :], gamma, idens, ixmom, iymom, iener)    

            if fmin * fmax > 0.0:
                pmax = min(pmax*1.0e2, 1.0)

            if fmin * fmax > 0.0:
                q[i, j, ip] = max((gamma-1.0)*U[i, j, iener], smallp)
            else:
                q[i, j, ip] = brentq(pmin, pmax, U[i, j, :], gamma, idens, ixmom, iymom, iener)

            if (q[i, j, ip] != q[i, j, ip]) or \
                (q[i, j, ip]-1.0 == q[i, j, ip]) or \
                (abs(q[i, j, ip]) > 1.0e10):  # nan or infty alert
                q[i, j, ip] = max((gamma-1.0)*U[i, j, iener], smallp)

            q[i, j, ip] = max(q[i, j, ip], smallp)
            if abs(U[i, j, iener] + U[i, j, idens] + q[i, j, ip]) < 1.0e-5:
                q[i, j, iu] = U[i, j, ixmom]
                q[i, j, iv] = U[i, j, iymom]
            else:
                q[i, j, iu] = U[i, j, ixmom]/(U[i, j, iener] + U[i, j, idens] + q[i, j, ip])
                q[i, j, iv] = U[i, j, iymom]/(U[i, j, iener] + U[i, j, idens] + q[i, j, ip])

            # nan check
            if (q[i, j, iu] != q[i, j, iu]):
                q[i, j, iu] = 0.0

            if (q[i, j, iv] != q[i, j, iv]):
                q[i, j, iv] = 0.0

    W = 1.0/np.sqrt(1.0 - q[:, :, iu]**2 - q[:, :, iv]**2)

    q[:, :, irho] = U[:, :, idens] / W
    if naux > 0:
        for i in range(naux):
            q[:, :, ix+i] = U[:, :, irhox+i]/(q[:, :, irho] * W)
