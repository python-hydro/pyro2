import h5py
import importlib
import mesh.patch as patch
import mesh.boundary as bnd


def read(filename):

    if not filename.endswith(".h5"):
        filename += ".h5"

    with h5py.File(filename, "r") as f:
        
        # read the simulation information -- this only exists if the 
        # file was created as a simulation object
        try:
            solver_name = f.attrs["solver"]
            problem_name = f.attrs["problem"]
            t = f.attrs["time"]
            n = f.attrs["nsteps"]
        except KeyError:
            # this was just a patch written out
            solver_name = None

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

        if solver_name is not None:
            solver = importlib.import_module(solver_name)

            sim = solver.Simulation(solver_name, problem_name, None)
            sim.n = n
            sim.cc_data = myd
            sim.cc_data.t = t

            sim.read_extras(f)

    if solver_name is not None:
        return sim
    else:
        return myd

            
