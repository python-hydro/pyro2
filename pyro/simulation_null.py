import h5py

import pyro.mesh.boundary as bnd
import pyro.util.profile_pyro as profile
from pyro.mesh import patch
from pyro.util import msg


def grid_setup(rp, ng=1):
    nx = rp.get_param("mesh.nx")
    ny = rp.get_param("mesh.ny")

    try:
        xmin = rp.get_param("mesh.xmin")
    except KeyError:
        xmin = 0.0
        msg.warning("mesh.xmin not set, defaulting to 0.0")

    try:
        xmax = rp.get_param("mesh.xmax")
    except KeyError:
        xmax = 1.0
        msg.warning("mesh.xmax not set, defaulting to 1.0")

    try:
        ymin = rp.get_param("mesh.ymin")
    except KeyError:
        ymin = 0.0
        msg.warning("mesh.ymin not set, defaulting to 0.0")

    try:
        ymax = rp.get_param("mesh.ymax")
    except KeyError:
        ymax = 1.0
        msg.warning("mesh.ymax not set, defaulting to 1.0")

    try:
        grid_type = rp.get_param("mesh.grid_type")
    except KeyError:
        grid_type = "Cartesian2d"
        msg.warning("mesh.grid_type not set, defaulting to Cartesian2D")

    if grid_type == "Cartesian2d":
        create_grid = patch.Cartesian2d
    elif grid_type == "SphericalPolar":
        create_grid = patch.SphericalPolar
    else:
        raise ValueError("Unsupported grid type!")

    my_grid = create_grid(nx, ny,
                          xmin=xmin, xmax=xmax,
                          ymin=ymin, ymax=ymax,
                          ng=ng)

    return my_grid


def bc_setup(rp):

    # first figure out the BCs
    try:
        xlb_type = rp.get_param("mesh.xlboundary")
    except KeyError:
        xlb_type = "periodic"
        msg.warning("mesh.xlboundary is not set, defaulting to periodic")

    try:
        xrb_type = rp.get_param("mesh.xrboundary")
    except KeyError:
        xrb_type = "periodic"
        msg.warning("mesh.xrboundary is not set, defaulting to periodic")

    try:
        ylb_type = rp.get_param("mesh.ylboundary")
    except KeyError:
        ylb_type = "periodic"
        msg.warning("mesh.ylboundary is not set, defaulting to periodic")

    try:
        yrb_type = rp.get_param("mesh.yrboundary")
    except KeyError:
        yrb_type = "periodic"
        msg.warning("mesh.yrboundary is not set, defaulting to periodic")

    bc = bnd.BC(xlb=xlb_type, xrb=xrb_type,
                ylb=ylb_type, yrb=yrb_type)

    # if we are reflecting, we need odd reflection in the normal
    # directions for the velocity
    bc_xodd = bnd.BC(xlb=xlb_type, xrb=xrb_type,
                     ylb=ylb_type, yrb=yrb_type,
                     odd_reflect_dir="x")

    bc_yodd = bnd.BC(xlb=xlb_type, xrb=xrb_type,
                     ylb=ylb_type, yrb=yrb_type,
                     odd_reflect_dir="y")

    return bc, bc_xodd, bc_yodd


class NullSimulation:

    def __init__(self, solver_name, problem_name, problem_func, rp, *,
                 problem_finalize_func=None,
                 timers=None, data_class=patch.CellCenterData2d):
        """
        Initialize the Simulation object

        Parameters
        ----------
        solver_name : str
            The name of the solver we are using
        problem_name : str
            The descriptive name for the problem (used in output)
        problem_func : function
            The function to call to initialize the problem
        rp : RuntimeParameters object
            The runtime parameters for the simulation
        problem_finalize_func : function
            An (optional) function to call when the simulation is
            over.
        timers : TimerCollection object, optional
            The timers used for profiling this simulation.
        data_class : CellCenterData2d or FV2d
            The class that manages the data.
        """

        self.n = 0
        self.dt = -1.e33
        self.dt_old = -1.e33

        self.data_class = data_class

        try:
            self.tmax = rp.get_param("driver.tmax")
        except (AttributeError, KeyError):
            self.tmax = None

        try:
            self.max_steps = rp.get_param("driver.max_steps")
        except (AttributeError, KeyError):
            self.max_steps = None

        self.rp = rp
        self.cc_data = None
        self.particles = None

        self.SMALL = 1.e-12

        self.solver_name = solver_name
        self.problem_name = problem_name
        self.problem_func = problem_func
        self.problem_finalize = problem_finalize_func

        if timers is None:
            self.tc = profile.TimerCollection()
        else:
            self.tc = timers

        try:
            self.verbose = self.rp.get_param("driver.verbose")
        except (AttributeError, KeyError):
            self.verbose = 0

        self.n_num_out = 0

        # plotting
        self.cm = "viridis"

    def finished(self):
        """
        is the simulation finished based on time or the number of steps
        """
        return self.cc_data.t >= self.tmax or self.n >= self.max_steps

    def do_output(self):
        """
        is it time to output?
        """
        dt_out = self.rp.get_param("io.dt_out")
        n_out = self.rp.get_param("io.n_out")
        do_io = self.rp.get_param("io.do_io")

        is_time = self.cc_data.t >= (self.n_num_out + 1)*dt_out or self.n % n_out == 0
        if is_time and do_io == 1:
            self.n_num_out += 1
            return True
        return False

    def initialize(self):
        pass

    def method_compute_timestep(self):
        """
        the method-specific timestep code
        """

    def compute_timestep(self):
        """
        a generic wrapper for computing the timestep that respects the
        driver parameters on timestepping
        """

        init_tstep_factor = self.rp.get_param("driver.init_tstep_factor")
        max_dt_change = self.rp.get_param("driver.max_dt_change")
        fix_dt = self.rp.get_param("driver.fix_dt")

        # get the timestep
        if fix_dt > 0.0:
            self.dt = fix_dt
        else:
            self.method_compute_timestep()
            if self.n == 0:
                self.dt = init_tstep_factor*self.dt
            else:
                self.dt = min(max_dt_change*self.dt_old, self.dt)
            self.dt_old = self.dt

        if self.cc_data.t + self.dt > self.tmax:
            self.dt = self.tmax - self.cc_data.t

    def preevolve(self):
        """
        Do any necessary evolution before the main evolve loop.  This
        is not needed for advection
        """

    def evolve(self):

        # increment the time
        self.cc_data.t += self.dt
        self.n += 1

    def dovis(self):
        pass

    def finalize(self):
        """
        Do any final clean-ups for the simulation and call the problem's
        finalize() method.
        """

        if self.problem_finalize:
            self.problem_finalize()

    def write(self, filename):
        """
        Output the state of the simulation to an HDF5 file for plotting
        """

        if not filename.endswith(".h5"):
            filename += ".h5"

        with h5py.File(filename, "w") as f:

            # main attributes
            f.attrs["solver"] = self.solver_name
            f.attrs["problem"] = self.problem_name
            f.attrs["time"] = self.cc_data.t
            f.attrs["nsteps"] = self.n

            self.cc_data.write_data(f)
            if self.particles is not None:
                self.particles.write_particles(f)
            self.rp.write_params(f)
            self.write_extras(f)

    def write_extras(self, f):
        """
        write out any extra simulation-specific stuff
        """

    def read_extras(self, f):
        """
        read in any simulation-specific data from an h5py file object f
        """
