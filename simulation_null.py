import importlib
import mesh.boundary as bnd
import mesh.patch as patch
from util import profile

def grid_setup(rp, ng=1):
    nx = rp.get_param("mesh.nx")
    ny = rp.get_param("mesh.ny")

    xmin = rp.get_param("mesh.xmin")
    xmax = rp.get_param("mesh.xmax")
    ymin = rp.get_param("mesh.ymin")
    ymax = rp.get_param("mesh.ymax")

    my_grid = patch.Grid2d(nx, ny,
                           xmin=xmin, xmax=xmax,
                           ymin=ymin, ymax=ymax, ng=ng)
    return my_grid


def bc_setup(rp):

    # first figure out the BCs
    xlb_type = rp.get_param("mesh.xlboundary")
    xrb_type = rp.get_param("mesh.xrboundary")
    ylb_type = rp.get_param("mesh.ylboundary")
    yrb_type = rp.get_param("mesh.yrboundary")

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


class NullSimulation(object):

    def __init__(self, solver_name, problem_name, rp, timers=None):
        """
        Initialize the Simulation object

        Parameters
        ----------
        problem_name : str
            The name of the problem we wish to run.  This should
            correspond to one of the modules in advection/problems/
        rp : RuntimeParameters object
            The runtime parameters for the simulation
        timers : TimerCollection object, optional
            The timers used for profiling this simulation
        """

        self.n = 0
        self.dt = -1.e33

        try: self.tmax = rp.get_param("driver.tmax")
        except:
            self.tmax = None

        try: self.max_steps = rp.get_param("driver.max_steps")
        except:
            self.max_steps = None

        self.rp = rp
        self.cc_data = None

        self.SMALL = 1.e-12

        self.solver_name = solver_name
        self.problem_name = problem_name

        if timers == None:
            self.tc = profile.TimerCollection()
        else:
            self.tc = timers

        try: self.verbose = self.rp.get_param("driver.verbose")
        except:
            self.verbose = None

        self.n_num_out = 0


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

        is_time = self.cc_data.t >= (self.n_num_out + 1)*dt_out or self.n%n_out == 0
        if is_time and do_io == 1:
            self.n_num_out += 1
            return True
        else:
            return False


    def initialize(self):
        pass


    def compute_timestep(self):
        pass


    def preevolve(self):
        """
        Do any necessary evolution before the main evolve loop.  This
        is not needed for advection
        """
        pass


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
        # there should be a cleaner way of doing this
        solver = importlib.import_module(self.solver_name)
        problem = importlib.import_module("{}.problems.{}".format(self.solver_name, self.problem_name))

        problem.finalize()
