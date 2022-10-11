# test the boundary fill routine


import numpy as np

import mesh.boundary as bnd
import mesh.patch


def doit():

    myg = mesh.patch.Grid2d(4, 4, ng=2, xmax=1.0, ymax=1.0)

    mydata = mesh.patch.CellCenterData2d(myg, dtype=np.int)

    bco = bnd.BC(xlb="outflow", xrb="outflow",
                 ylb="outflow", yrb="outflow")
    mydata.register_var("outflow", bco)

    bcp = bnd.BC(xlb="periodic", xrb="periodic",
                 ylb="periodic", yrb="periodic")
    mydata.register_var("periodic", bcp)

    bcre = bnd.BC(xlb="reflect-even", xrb="reflect-even",
                  ylb="reflect-even", yrb="reflect-even")
    mydata.register_var("reflect-even", bcre)

    bcro = bnd.BC(xlb="reflect-odd", xrb="reflect-odd",
                  ylb="reflect-odd", yrb="reflect-odd")
    mydata.register_var("reflect-odd", bcro)

    mydata.create()

    a = mydata.get_var("outflow")

    for i in range(myg.ilo, myg.ihi+1):
        for j in range(myg.jlo, myg.jhi+1):
            a[i, j] = (i-myg.ilo) + 10*(j-myg.jlo) + 1

    b = mydata.get_var("periodic")
    c = mydata.get_var("reflect-even")
    d = mydata.get_var("reflect-odd")

    b[:, :] = a[:, :]
    c[:, :] = a[:, :]
    d[:, :] = a[:, :]

    mydata.fill_BC("outflow")
    mydata.fill_BC("periodic")
    mydata.fill_BC("reflect-even")
    mydata.fill_BC("reflect-odd")

    print("outflow")
    mydata.pretty_print("outflow")

    print(" ")
    print("periodic")
    mydata.pretty_print("periodic")

    print(" ")
    print("reflect-even")
    mydata.pretty_print("reflect-even")

    print(" ")
    print("reflect-odd")
    mydata.pretty_print("reflect-odd")


if __name__ == "__main__":
    doit()
