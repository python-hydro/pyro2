"""This manages the reading of the HDF5 output files for pyro.

"""
import importlib

import h5py
import mesh.patch as patch
import mesh.boundary as bnd
import particles.particles as particles


def read_bcs(f):
    """read in the boundary condition record from the HDF5 file"""
    try:
        gb = f["BC"]
    except KeyError:
        return None
    else:
        BCs = {}
        for name in gb:
            BCs[name] = gb[name]

        return BCs


def read(filename):
    """read an HDF5 file and recreate the simulation object that holds the
    data and state of the simulation.

    """
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

        # sometimes problems define custom BCs -- at the moment, we
        # are going to assume that these always map to BC.user.  We
        # need to read these in now, since the variable creation
        # requires it.
        custom_bcs = read_bcs(f)
        if custom_bcs is not None:
            if solver_name in ["compressible_fv4", "compressible_rk", "compressible_sdc"]:
                bc_solver = "compressible"
            else:
                bc_solver = solver_name
            bcmod = importlib.import_module("{}.{}".format(bc_solver, "BC"))
            for name in custom_bcs:
                bnd.define_bc(name, bcmod.user, is_solid=custom_bcs[name])

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
            v.v()[:, :] = data[:, :]

        # restore the particle data
        try:
            gparticles = f["particles"]
            particle_data = gparticles["particle_positions"]
            init_data = gparticles["init_particle_positions"]

            my_particles = particles.Particles(myd, None, len(particle_data),
                                            "array", particle_data, init_data)
        except KeyError:
            my_particles = None

        if solver_name is not None:
            solver = importlib.import_module(solver_name)

            sim = solver.Simulation(solver_name, problem_name, None)
            sim.n = n
            sim.cc_data = myd
            sim.cc_data.t = t
            sim.particles = my_particles

            sim.read_extras(f)

    if solver_name is not None:
        return sim

    return myd
