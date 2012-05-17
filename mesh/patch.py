import numpy

    
class cellCentered2d:
    """
    the 2-d patch class.  The patch object will contain the coordinate
    information (at various centerings) and the state data itself
    (cell-centered).

    A basic (1-d) representation of the layout is:

    |     |      |     X     |     |      |     |     X     |      |     |
    +--*--+- // -+--*--X--*--+--*--+- // -+--*--+--*--X--*--+- // -+--*--+
       0          ng-1    ng   ng+1         ... ng+nx-1 ng+nx      2ng+nx-1

                         ilo                      ihi
    
    |<- ng guardcells->|<---- nx interior zones ----->|<- ng guardcells->|

    The '*' marks the data locations.


    a patch is built in a multi-step process before it can be used:

    create the patch object:

        myPatch = patch.cellCentered2d(nx, ny)

    register any variables that w eexpect to live on this patch

        myPatch.registerVar('density')
        myPatch.registerVar('x-momentum')
        myPatch.registerVar('y-momentum')
        ...

    finally, initialize the patch

        myPatch.init()

    This last step actually allocates the storage for the state
    variables.  Once this is done, the patch is considered to be
    locked.  New variables cannot be added.
    """

    #-------------------------------------------------------------------------
    # __init__
    #-------------------------------------------------------------------------
    def __init__ (self, nx, ny, ng=1, \
                  xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, \
                  xlb = "reflect", xrb = "reflect", \
                  ylb = "reflect", yrb = "reflect"):

        """
        The class constructor function.

        The only data that we require is the number of points that
        make up the mesh.

        We optionally take the extrema of the domain (assume it is
        [0,1]x[0,1]), number of ghost cells (assume 1), and the
        type of boundary conditions (assume reflecting).
        """

        self.allVars = []
        self.nAllVars = 0
        
        self.vars = []
        self.nvar = 0

        self.species = []
        self.nspecies = 0

        # size of grid
        self.nx = nx
        self.ny = ny
        self.ng = ng

        self.qx = 2*ng+nx
        self.qy = 2*ng+ny

        
        # domain extrema
        self.xmin = xmin
        self.xmax = xmax

        self.ymin = ymin
        self.ymax = ymax

        # domain boundary conditions
        self.xlb = xlb
        self.xrb = xrb

        self.ylb = ylb
        self.yrb = yrb

        # compute the indices of the block interior (excluding guardcells)
        self.ilo = ng
        self.ihi = ng+nx-1

        self.jlo = ng
        self.jhi = ng+ny-1

        # define the coordinate information at the left, center, and right
        # zone positions
        self.dx = (xmax - xmin)/nx

        self.xl = (numpy.arange(nx+2*ng) - ng)*self.dx + xmin
        self.xr = (numpy.arange(nx+2*ng) + 1.0 - ng)*self.dx + xmin
        self.x = 0.5*(self.xl + self.xr)

        self.dy = (ymax - ymin)/ny

        self.yl = (numpy.arange(ny+2*ng) - ng)*self.dy + ymin
        self.yr = (numpy.arange(ny+2*ng) + 1.0 - ng)*self.dy + ymin
        self.y = 0.5*(self.yl + self.yr)

        x2d = numpy.repeat(self.x, 2*self.ng+self.ny)
        x2d.shape = (2*self.ng+self.nx, 2*self.ng+self.ny)
        self.x2d = x2d

        y2d = numpy.repeat(self.y, 2*self.ng+self.nx)
        y2d.shape = (2*self.ng+self.ny, 2*self.ng+self.nx)
        y2d = numpy.transpose(y2d)
        self.y2d = y2d
                                                                        

        # time information
        self.time = 0.0
        self.nstep = 0


    def registerVar(self, name):
        """ register a variable with patch object """

        self.vars.append(name)
        self.nvar += 1


    def registerSpecies(self, name):
        """ register a species variable with patch object """

        self.species.append(name)
        self.nspecies += 1


    def init(self):
        """
        called after all the variables are registered and allocates
        the storage for the unknowns
        """

        self.allVars = self.vars + self.species
        self.nAllVars = self.nvar + self.nspecies
        
        self.data = numpy.zeros((self.nAllVars, 2*self.ng+self.nx, 2*self.ng+self.ny),
                                dtype=numpy.float64)

        
    def __str__(self):
        """ print out some basic information about the patch object """
        string = "2-d patch: nx = " + `self.nx` + ', ny = ' + `self.ny` + ', ng = ' + `self.ng` + "\n" + \
                 "           nAllVars = " + `self.nAllVars` + "\n" + \
                 "           all variables: " + `self.allVars`
        return string
        

    def getVarPtr(self, name):

        n = self.allVars.index(name)
        # we have a choice here, we can return in essence, a pointer to the
        # variable in the main self.data array, or, by commenting out the
        # next line, return a copy, which would require an explicit put
        # method.
        #return self.data[:,:,n].copy()
        return self.data[n,:,:]


    def getVarPtrByIndex(self, index):
        return self.data[index,:,:]


    def zero(self, name):
        n = self.allVars.index(name)
        self.data[n,:,:] = 0.0
        
                
    def fillBC(self):
        """ fill the boundary conditions """
    
        # do the boundaries one at a time.  The single grid case
        # is easy -- every boundary is a physical boundary of the
        # domain, so there is not much trickery to do

        # we do periodic, outflow, reflect, and in the y-direction,
        # HSE boundaries.
        
        # lower X boundary
        if (self.xlb == "reflect" or self.xlb == "outflow"):
            n = 0
            while n < self.nvar:

                i = 0
                while i < self.imin:
                    self.data[n,i,:] = self.data[n,self.imin,:]
                    
                    i += 1
                
                n+= 1

            # if we are reflecting, actually reflect the either the
            # x-momentum or x-velocity (different solvers use different
            # variable types
            if (self.xlb == "reflect"):
                try:
                    npx = self.allVars.index("x-momentum")
                except ValueError:
                    npx = -1

                try:
                    nvx = self.allVars.index("x-velocity")
                except ValueError:
                    nvx = -1

                i = 0
                while i < self.imin:
                    if npx >= 0:
                        self.data[npx,i,:] = -self.data[npx,2*self.ng-i-1,:]

                    if nvx >= 0:
                        self.data[nvx,i,:] = -self.data[nvx,2*self.ng-i-1,:]
                        
                    i += 1


        elif (self.xlb == "periodic"):
            n = 0
            while n < self.nvar:

                i = 0
                while i < self.imin:
                    self.data[n,i,:] = self.data[n,self.imax-self.ng+i+1,:]

                    i += 1

                n += 1
            

        # upper X boundary
        if (self.xrb == "reflect" or self.xrb == "outflow"):
            n = 0
            while n < self.nvar:

                i = self.imax+1
                while i < self.nx+2*self.ng:
                    self.data[n,i,:] = self.data[n,self.imax,:]

                    i += 1
                
                n+= 1

            # if we are reflecting, actually reflect the x-momentum now
            if (self.xrb == "reflect"):
                try:
                    npx = self.allVars.index("x-momentum")
                except ValueError:
                    npx = -1

                try:
                    nvx = self.allVars.index("x-velocity")
                except ValueError:
                    nvx = -1

                i = 0
                while i < self.ng:
                    i_bnd = self.imax+1+i
                    i_src = self.imax-i
                    
                    if npx >= 0:
                        self.data[npx,i_bnd,:] = -self.data[npx,i_src,:]

                    if nvx >= 0:
                        self.data[nvx,i_bnd,:] = -self.data[nvx,i_src,:]
                        
                    i += 1


        elif (self.xrb == "periodic"):
            n = 0
            while n < self.nvar:

                i = self.imax+1
                while i < 2*self.ng + self.nx:
                    self.data[n,i,:] = self.data[n,i-self.imax-1+self.ng,:]

                    i += 1

                n += 1
                

        # lower Y boundary
        if (self.ylb == "reflect" or self.ylb == "outflow"):
            n = 0
            while n < self.nvar:

                j = 0
                while j < self.jmin:
                    self.data[n,:,j] = self.data[n,:,self.jmin]
                    
                    j += 1
                
                n+= 1

            # if we are reflecting, actually reflect the y-momentum now
            if (self.ylb == "reflect"):
                try:
                    npy = self.allVars.index("y-momentum")
                except ValueError:
                    npy = -1

                try:
                    nvy = self.allVars.index("y-velocity")
                except ValueError:
                    nvy = -1

                j = 0
                while j < self.jmin:
                    if npy >= 0:
                        self.data[npy,:,j] = -self.data[npy,:,2*self.ng-j-1]

                    if nvy >= 0:
                        self.data[nvy,:,j] = -self.data[nvy,:,2*self.ng-j-1]
                        
                    j += 1


        elif (self.ylb == "periodic"):
            n = 0
            while n < self.nvar:

                j = 0
                while j < self.jmin:
                    self.data[n,:,j] = self.data[n,:,self.jmax-self.ng+j+1]

                    j += 1

                n += 1
                

        # upper Y boundary
        if (self.yrb == "reflect" or self.yrb == "outflow"):
            n = 0
            while n < self.nvar:

                j = self.jmax+1
                while j < self.ny+2*self.ng:
                    self.data[n,:,j] = self.data[n,:,self.jmax]
                    
                    j += 1
                
                n+= 1

            # if we are reflecting, actually reflect the y-momentum now
            if (self.yrb == "reflect"):
                try:
                    npy = self.allVars.index("y-momentum")
                except ValueError:
                    npy = -1

                try:
                    nvy = self.allVars.index("y-velocity")
                except ValueError:
                    nvy = -1

                j = 0
                while j < self.ng:
                    j_bnd = self.jmax+1+j
                    j_src = self.jmax-j
                    
                    if npy >= 0:
                        self.data[npy,:,j_bnd] = -self.data[npy,:,j_src]

                    if nvy >= 0:
                        self.data[nvy,:,j_bnd] = -self.data[nvy,:,j_src]                      
                    
                    j += 1

        
        elif (self.yrb == "periodic"):
            n = 0
            while n < self.nvar:

                j = self.jmax+1
                while j < 2*self.ng + self.ny:
                    self.data[n,:,j] = self.data[n,:,j-self.jmax-1+self.ng]

                    j += 1

                n += 1

