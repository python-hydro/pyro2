import h5py

import mesh.patch as patch
import mesh.boundary as bnd


f = h5py.File("test.h5", "r")

# read the simulation information
solver = f.attrs["solver"]
problem = f.attrs["problem"]

t = f.attrs["time"]
n = f.attrs["nsteps"]

# read in the grid info and create our grid
grid = f["grid"].attrs

myg = patch.Grid2d(grid["nx"], grid["ny"], ng=grid["ng"], 
                   xmin=grid["xmin"], xmax=grid["xmax"], 
                   ymin=grid["ymin"], ymax=grid["ymax"])



# read in the variable info -- start by getting the names
gs = f["state"]
names = []
for n in gs:
    names.append(n)

# create the CellCenterData2d object
myd = patch.CellCenterData2d(myg)

for n in names:
    grp = gs[n]
    bc = bnd.BC(xlb=grp.attrs["xlb"], xrb=grp.attrs["xrb"],
                ylb=grp.attrs["ylb"], yrb=grp.attrs["yrb"])
    myd.register_var(n, bc)

myd.create()

# auxillary data
for k in f["aux"].attrs:
    myd.set_aux(k, f["aux"].attrs[k])

# restore the variable data
for n in names:
    grp = gs[n]
    data = grp["data"]

    v = myd.get_var(n)
    v.v()[:,:] = data[:,:]


print(myd)



