import mesh.reconstruction as reconstruction

def fluxes(my_data, rp, dt):
    """Construct the fluxes through the interfaces for the linear advection
    equation:

      a  + u a  + v a  = 0
       t      x      y

    We use a fourth-order Godunov method to construct the interface
    states, using Runge-Kutta integration.  Since this is 4th-order,
    we need to be aware of the difference between a face-average and
    face-center for the fluxes.

    In the pure advection case, there is no Riemann problem we need to
    solve -- we just simply do upwinding.  So there is only one 'state'
    at each interface, and the zone the information comes from depends
    on the sign of the velocity.

    Our convection is that the fluxes are going to be defined on the
    left edge of the computational zones


     |             |             |             |
     |             |             |             |
    -+------+------+------+------+------+------+--
     |     i-1     |      i      |     i+1     |

              a_l,i  a_r,i   a_l,i+1


    a_r,i and a_l,i+1 are computed using the information in
    zone i,j.

    Parameters
    ----------
    my_data : FV object
        The data object containing the grid and advective scalar that
        we are advecting.
    rp : RuntimeParameters object
        The runtime parameters for the simulation
    dt : float
        The timestep we are advancing through.
    scalar_name : str
        The name of the variable contained in my_data that we are
        advecting

    Returns
    -------
    out : ndarray, ndarray
        The fluxes averaged over the x and y faces

    """

    myg = my_data.grid

    a = my_data.get_var("density")

    # get the advection velocities
    u = rp.get_param("advection.u")
    v = rp.get_param("advection.v")


    # interpolate cell-average a to face-averaged a on interfaces in each
    # dimension


    # calculate the face-centered a using the transverse Laplacian


    # compute the face-averaed fluxes 

    return F_x, F_y
