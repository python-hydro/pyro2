# pyro
*A simple python-based tutorial on computational methods for hydrodynamics*


The latest version of pyro is always available at:

https://github.com/zingale/pyro2

The project webpage, where you'll find documentation, plots, notes,
etc. is here:

http://bender.astro.sunysb.edu/hydro_by_example/



## Getting Started

  - There are a few steps to take to get things running. You need to
     make sure you have numpy, f2py, and matplotlib installed. On a
     Fedora system, this can be accomplished by doing:

       yum install numpy numpy-f2py python-matplotlib python-matplotlib-tk

  - You also need to make sure gfortran is present on you system. On
     a Fedora system, it can be installed as: 

       yum install gcc-gfortran 

  - Not all matplotlib backends allow for the interactive plotting as
     pyro is run. One that does is the TkAgg backend. This can be made
     the default by creating a file ~/.matplotlib/matplotlibrc with
     the content:

       backend: TkAgg 

     You can check what backend is your current default in python via: 

       import matplotlib.pyplot 
       print matplotlib.pyplot.get_backend() 

  - The remaining steps are: 

      * Set the PYTHONPATH environment variable to point to the pyro2/ directory.

      * Build the Fortran source. In pyro2/ type 

          ./mk.sh 

      * Run a quick test of the advection solver: 

          ./pyro.py advection smooth inputs.smooth 

        you should see a graphing window pop up with a smooth pulse
        advecting diagonally through the periodic domain.


## Working With Data

  Some problems write a report at the end of the simulation specifying
  the analysis routines that can be used with their data.

  The plot.py script in the pyro2/ directory can be used to directly
  visualize the data from a .pyro output file.


## Getting Help

  Join the mailing list to say up-to-date:

  http://bender.astro.sunysb.edu/mailman/listinfo/pyro-help

