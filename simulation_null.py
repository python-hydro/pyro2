import mesh.patch as patch

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
    

class NullSimulation:

    def __init__(self, problem_name, rp, timers=None):
        """
        Initialize the Simulation object for linear advection.

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

        self.rp = rp
        self.cc_data = None

        self.SMALL = 1.e-12

        self.problem_name = problem_name

        if timers == None:
            self.tc = profile.TimerCollection()
        else:
            self.tc = timers

    def finished(self):
        pass
    
    def initialize(self):
        pass

    def timestep(self):
        pass

    def preevolve(self):
        """
        Do any necessary evolution before the main evolve loop.  This
        is not needed for advection
        """
        pass

    def evolve(self, dt):
        pass

    def dovis(self):
        pass

    def finalize(self):
        """
        Do any final clean-ups for the simulation and call the problem's
        finalize() method.
        """
        exec(self.problem_name + '.finalize()')
