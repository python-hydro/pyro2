import h5py

import mesh.patch as patch
import mesh.boundary as bnd

myg = patch.Grid2d(8, 16, xmax=1.0, ymax=2.0)
mydata = patch.CellCenterData2d(myg)

bc = bnd.BC()

mydata.register_var("a", bc)
mydata.register_var("b", bc)

mydata.create()

mydata.set_aux("1", 1)
mydata.set_aux("2", 2)

# we need to figure out how to do the boundary conditions


f = h5py.File("test.h5", "w")


# main attributes
f.attrs["solver"] = "compressible"
f.attrs["problem"] = "Sedov"

f.attrs["time"] = mydata.t
f.attrs["nsteps"] = 0


# auxillary data
grp_aux = f.create_group("aux")

for a in mydata.aux:
    grp_aux.attrs[a] = mydata.aux[a]


# grid infomation
grp_grid = f.create_group("grid")

grp_grid.attrs["nx"] = myg.nx
grp_grid.attrs["ny"] = myg.ny
grp_grid.attrs["ng"] = myg.ng

grp_grid.attrs["xmin"] = myg.xmin
grp_grid.attrs["xmax"] = myg.xmax
grp_grid.attrs["ymin"] = myg.ymin
grp_grid.attrs["ymax"] = myg.ymax


# data
grp_data = f.create_group("state")

for n in range(mydata.nvar):
    grp_var = grp_data.create_group(mydata.names[n])
    grp_var.create_dataset("data",
                           data=mydata.get_var_by_index(n).v())

    grp_var.attrs["xlb"] = mydata.BCs[mydata.names[n]].xlb
    grp_var.attrs["xrb"] = mydata.BCs[mydata.names[n]].xrb
    grp_var.attrs["ylb"] = mydata.BCs[mydata.names[n]].ylb
    grp_var.attrs["yrb"] = mydata.BCs[mydata.names[n]].yrb

f.close()
