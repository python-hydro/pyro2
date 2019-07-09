import numpy as np


def compute_density(my_data, ivars):
    U = my_data.data
    h = my_data.get_aux("h")

    U[:, ivars.irho] = 0

    for i in range(my_data.np):

        C =  4 / (np.pi * h**8)
        U[i, ivars.irho] += 4 * U[i, ivars.im] / (np.pi * h**2)

        for j in range(i+1):
            r_ij = U[i, ivars.ix:ivars.iy + 1] - U[j, ivars.ix:ivars.iy + 1]
            z = h**2 - np.dot(r_ij, r_ij)

            if z > 0:
                U[i, ivars.irho] += C * U[i, ivars.im] * z**3
                U[j, ivars.irho] += C * U[j, ivars.im] * z**3

def compute_acceleration(my_data, ivars):

    compute_density(my_data, ivars)

    U = my_data.data
    h = my_data.get_aux("h")
    rho0 = my_data.get_aux("rho0")
    mu = my_data.get_aux("mu")
    k = my_data.get_aux("k")

    U[:, ivars.iax] = 0
    U[:, ivars.iay] = my_data.get_aux("g")

    for i in range(my_data.np):
        for j in range(i):
            r_ij = U[i, ivars.ix:ivars.iy + 1] - U[j, ivars.ix:ivars.iy + 1]
            v_ij = U[i, ivars.iu:ivars.iv + 1] - U[j, ivars.iu:ivars.iv + 1]
            q = np.sqrt(np.dot(r_ij, r_ij)) / h
            if q > 0:
                f_ij = (1 - q) * (15 * k * (U[i, ivars.irho] +
                                            U[j, ivars.irho] - 2 * rho0) *
                                  (1 - q) / q * r_ij - 40 * mu * v_ij)
            else:
                f_ij = - 40 * mu * v_ij

            U[i, ivars.iax:ivars.iay + 1] += f_ij * U[j, ivars.im] / \
                (np.pi * h**4 * U[j, ivars.irho] * U[i, ivars.irho])

            U[j, ivars.iax:ivars.iay + 1] -= f_ij * U[i, ivars.im] / \
                (np.pi * h**4 * U[j, ivars.irho] * U[i, ivars.irho])
