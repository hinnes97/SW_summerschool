import numpy as np
from . import forcing
from .initial_conditions import vortex, gaussian, plane_gravity_wave, barotropic_jet, rossby

class ArakawaCGrid:

    def __init__(self, N, Ro, bc='periodic', force =None, ic=None, udrag=None,
                 hdrag = None, beta=0.):
        """Creates Arakawa C-grid object with 2*N x points and N y-points"""

        # Set resolution
        self.d = 1/N

        # Make equatorial beta plane 2 times longer than tall (planetary scale)
        self.Nx = 2*N
        self.Ny = N

        # Edge y (with ghost)
        self.ye = np.linspace(-0.5-self.d*2, 0.5+self.d*2, self.Ny+5)
        # Edge x (with ghost)
        self.xe = np.linspace(-1-self.d*2, 1.+self.d*2, self.Nx+5)
        # Mid x (with ghost)
        self.xm = np.linspace(-1-self.d*1.5, 1.+self.d*1.5, self.Nx+4)
        # Mid y (with ghost)
        self.ym = np.linspace(-0.5-self.d*1.5, 0.5+self.d*1.5,self.Ny+4)

        # Set y boundary condition
        self.ybc = bc
        
        # Define C-grid variables
        # j+1    q-----v-----q-----v-----q
        #        |           |           |
        # j+1/2  u     h     u     h     u
        #        |           |           |
        # j      q-----v-----q-----v-----q
        #        |           |           |
        # j-1/2  u     h     u     h     u
        #        |           |           |
        # j-1    q-----v-----q-----v-----q
        #
        #        i   i-1/2.  i   i+1/2  i+1

        hgrid_slice = (slice(2,-2,1), slice(2,-2,1))
        ugrid_slice = (slice(2,-3,1), slice(2,-2,1))

        if self.ybc=='free-slip-wall':
            vgrid_slice=(slice(2,-2,1), slice(2,-2,1))
            qgrid_slice=(slice(1,-2,1), slice(1,-1,1))
        else:
            vgrid_slice=(slice(2,-2,1), slice(2,-3,1))
            qgrid_slice=(slice(1,-2,1), slice(1,-2,1))
        
        self.realslice = {
            "h": hgrid_slice,
            "u": ugrid_slice,
            "v": vgrid_slice,
            "K": hgrid_slice,
            "q": qgrid_slice,
            "E":()
        }

        self.progslice={
            "h": hgrid_slice,
            "u": (slice(2,-2,1), slice(2,-2,1)),
            "v": (slice(2,-2,1), slice(2,-2,1))
        }

        # Set up grids, including halo points
        # Main variables u,v,h
        self.u = np.zeros((self.Nx+5, self.Ny+4))
        self.v = np.zeros((self.Nx+4, self.Ny+5))
        self.h = np.zeros((self.Nx+4, self.Ny+4))

        # Gravity parameter
        self.Ro = Ro
        self.beta = beta
        
        # Helper variables
        self.ustar = np.zeros((self.Nx+3, self.Ny+2)) # mass flux on u points
        self.vstar = np.zeros((self.Nx+2, self.Ny+3)) # mass flux on v points

        # Potential vorticity and kinetic energy
        self.q = np.zeros((self.Nx+3, self.Ny+3))
        self.K = np.zeros((self.Nx+2, self.Ny+2))

        # Total energy
        self.E = 0.0

        # Coefficient grids (eq 3.34 Arakawa and Lamb)
        self.eps = np.zeros((self.Nx+2, self.Ny+2))
        self.phi = np.zeros((self.Nx+2, self.Ny+2))
        self.alp = np.zeros((self.Nx+1, self.Ny+2))
        self.bet = np.zeros((self.Nx+1, self.Ny+2))
        self.gam = np.zeros((self.Nx+1, self.Ny+2))
        self.det = np.zeros((self.Nx+1, self.Ny+2))

        # Set dissipation
        self.udrag = udrag
        self.hdrag = hdrag

        # Set forcing
        self.forcing_params = force
        if force["type"] == "Gill":
            self.forcing = forcing.gill_forcing
        else:
            self.forcing = forcing.null_forcing

        # Initial condition
        self.ic = ic

        if self.ic["type"] == "gaussian":
            gaussian(self, self.ic["amp"],
                     self.ic["width"],
                     self.ic["x0"],
                     self.ic["y0"])
            
        elif self.ic["type"] == "vortex":
            vortex(self, self.ic["amp"], self.ic["width"])
        elif self.ic["type"] == "plane_gravity_wave":
            print('plane wave ic')
            plane_gravity_wave(self, self.ic["amp"], self.ic["width"])
        elif self.ic["type"] == "barotropic_jet":
            barotropic_jet(self, self.ic["amp"], self.ic["width"])
        elif self.ic["type"] == 'rossby':
            rossby(self, self.ic["amp"])
        else:
            # Don't do anything
            self.h = np.ones_like(self.h)

        self.apply_bcs()
        self.update_diagnostics()
        self.io_diagnostics()
        
        # Tell IO what grid each variable is on
        self.vargrid = {
            "h": ("time", "xmid", "ymid"),
            "u": ("time", "xedge", "ymid"),
            "v": ("time", "xmid", "yedge"),
            "K": ("time", "xmid", "ymid"),
            "q": ("time", "xedge", "yedge"),
            "E": ("time")
        }

        # What type of interpolation needs to be done on output 
        self.interptype = {
            "h": None,
            "u": "x",
            "v": "y",
            "q": "xy",
            "K": None,
            "E": None
        }

        # Which variables are defined on x edges
        self.xe_vars = ("q", "u")
        # Same, for y
        self.ye_vars = ("q", "v")

    def __iter__(self):
        """Yield ``(name, array)`` pairs for ``u``, ``v`` and ``h`` in order.
        
        Arrays include halo points and are references to the grid state,
        so modifying a yielded array also modifies the model."""
        yield "u", self.u
        yield "v", self.v
        yield "h", self.h

    def __getitem__(self, key):
        """Return the grid attribute named by the string ``key``.
        
        Arrays are returned by reference, including halo points. Scalar
        diagnostics can also be accessed. Raises AttributeError when the
        requested attribute does not exist."""
        return getattr(self, key)
        
    def calc_coeffs(self):
        """Updates coefficients on the Arakawa C-Grid"""

        q = self.q
        
        # These are defined on the h grid, inc. ghost cells
        self.eps = 1./24. * (q[1:,1:] + q[:-1, 1:]  - q[:-1, :-1] - q[1:,:-1])
        self.phi = 1./24. *  (-q[1:,1:] + q[:-1,1:]  + q[:-1,:-1] - q[1:,:-1])

        # All defined at u points (i,j+1/2), including y ghosts but not with x ghost
        self.alp = 1./24. * (2*q[2:,1:] + q[1:-1, 1:] + 2*q[1:-1,:-1] + q[2:,:-1])
        self.bet = 1./24. * (q[1:-1,1:] + 2*q[:-2,1:] + q[:-2,:-1] + 2*q[1:-1,:-1])
        self.gam = 1./24. * (2*q[1:-1,1:] + q[:-2,1:] + 2*q[:-2,:-1] + q[1:-1,:-1])
        self.det = 1./24 *  (q[2:,1:] + 2*q[1:-1, 1:] + q[1:-1,:-1] + 2*q[2:,:-1])

    def calc_ustars(self):
        """Update staggered mass fluxes from adjacent height averages.
        
        Sets ``ustar`` to averaged height times zonal velocity, with shape
        ``(Nx + 3, Ny + 2)``, and ``vstar`` to averaged height times
        meridional velocity, with shape ``(Nx + 2, Ny + 3)``. Requires
        current boundary values in ``h``, ``u`` and ``v``. Returns None."""
        hu = 0.5*(self.h[:-1,1:-1] + self.h[1:,1:-1])
        hv = 0.5*(self.h[1:-1,:-1] + self.h[1:-1,1:])

        self.ustar = hu*self.u[1:-1,1:-1]
        self.vstar = hv*self.v[1:-1,1:-1]

    def apply_bcs(self):
        """Update velocity and height boundary values and halos in place.
        
        The x direction is periodic. For ``free-slip-wall`` y boundaries,
        sets normal velocity to zero at the walls and in its halos, and
        extends adjacent interior height and zonal velocity into halos.
        Otherwise applies periodic y boundaries. Returns None."""
        if self.ybc == 'free-slip-wall':
            self.v[:,2] = 0.
            self.v[:,-3] = 0
            self.v[:,:2] = 0.
            self.v[:,-2:] = 0.

            self.h[:,:2] = self.h[:,2,None]
            self.h[:,-2:] = self.h[:,-3,None]
            self.u[:,:2] = self.u[:,2,None]
            self.u[:,-2:] = self.u[:,-3,None]

        else:
            # Periodic
            self.v[:,-3] = self.v[:,2]
            self.v[:,-2] = self.v[:,3]
            self.v[:,-1] = self.v[:,4]
            self.v[:, 0] = self.v[:,-5]
            self.v[:, 1] = self.v[:,-4]

            for var in [self.u, self.h]:
                var[:,-2] = var[:,2]
                var[:,-1] = var[:,3]
                var[:, 0] = var[:,-4]
                var[:, 1] = var[:,-3]


        # X direction, all periodic 
        self.u[-3,:] = self.u[2,:]
        self.u[-2,:] = self.u[3,:]
        self.u[-1,:] = self.u[4,:]
        self.u[0,:] = self.u[-5,:]
        self.u[1,:] = self.u[-4,:]
        
        for var in [self.h, self.v]:
            var[0,:] = var[-4,:]
            var[1,:] = var[-3,:]
            var[-2,:] = var[2,:]
            var[-1,:] = var[3,:]
        
    def coriolis(self):
        # Only equatorial beta-plane currently implemented, in non-dimensionalised form
        """Return the nondimensional meridional Coriolis profile.
        
        Returns
        -------
        numpy.ndarray
            Shape ``(Ny + 3,)``, evaluated as ``1 / Ro + beta * y`` on
            evenly spaced coordinates from -0.5 to 0.5. Broadcasts along
            the second axis of the potential-vorticity array.
        
        Notes
        -----
        Requires nonzero ``Ro``; no zero-rotation special case is applied."""
        y = np.linspace(-0.5, 0.5, self.Ny+3)
        return 1/self.Ro + self.beta*y
        
    def calc_vorticity(self):
        """Calculate the potential vorticity q = (zeta + f)/h
           Definition of variables Arakawa and Lamb eq. 3.11, 3.12"""

        # Relative vorticity defined on the q points
        zeta = 1./self.d * (self.u[1:-1,:-1] - self.u[1:-1,1:] + self.v[1:,1:-1] - self.v[:-1,1:-1])

        # Require h on q grid
        hq = 1./4.*(self.h[1:,1:] + self.h[:-1,1:] + self.h[1:,:-1] + self.h[:-1,:-1])

        f = self.coriolis()

        self.q = (zeta + f)/hq

    def calc_KE(self):
        """Kinetic energy at h grid (centres)"""
        
        usq = 0.5*((self.u**2)[1:-2,1:-1] + (self.u**2)[2:-1,1:-1])
        vsq = 0.5*((self.v**2)[1:-1,1:-2] + (self.v**2)[1:-1,2:-1])

        self.K = 0.5*(usq + vsq)

    def divergence(self):
        """"Calculating the divergence of the mass flux (Arakawa and Lamb, eq. 3.2)
            Do not bother calculating ghost cells, only needed to update h"""
        # dim ustar Nx+3, Ny+2, dim vstar Nx+2, Ny+3
        # want variable Nx, Ny
        return 1./self.d * (self.ustar[2:-1,1:-1] - self.ustar[1:-2,1:-1] + self.vstar[1:-1,2:-1] - self.vstar[1:-1, 1:-2])
       
    def dh_dt(self):
        """Return dh_dt from equation 3.1, AL80"""
        if self.hdrag["type"] == "rayleigh":
            drag = -self.h[self.realslice["h"]]/self.hdrag["tau_h"]
        else:
            drag = 0.
            
        return -self.divergence() + self.forcing(self) + drag


    def du_dt(self):
        """Return du_dt, equation 3.5 AL80
           Only return on main grid, not ghost points =  (Nx+1,Ny)"""

        #self.calc_coeffs()
        #ustar, vstar = self.calc_ustars()
        
        t1 = self.alp[:,1:-1]*self.vstar[1:,2:-1]
        t2 = self.bet[:,1:-1]*self.vstar[:-1,2:-1]
        t3 = self.gam[:,1:-1]*self.vstar[:-1,1:-2]
        t4 = self.det[:,1:-1]*self.vstar[1:,1:-2]
        t5 = -self.eps[1:,1:-1]*self.ustar[2:,1:-1]
        t6 = self.eps[:-1,1:-1]*self.ustar[:-2,1:-1]

        t7 = -1./self.d *(self.K[1:,1:-1] + self.h[2:-1,2:-2] - self.K[:-1,1:-1] - self.h[1:-2,2:-2])

        if self.udrag["type"] == "rayleigh":
            drag = -self.u[2:-2,2:-2]/self.udrag["tau_u"]
        else:
            drag = 0.0

        return t1 + t2 + t3 + t4 + t5 + t6 + t7 + drag
    
    def dv_dt(self):
        """Return dv_dt, equation 3.6 AL80
           Only return on main grid, not ghost points =  (Nx,Ny+1)"""
        # Coeffs on grid Nx+1,Ny+2

        t1 = -self.gam[1:,1:]*self.ustar[2:-1,1:]
        t2 = -self.det[:-1,1:]*self.ustar[1:-2,1:]
        t3 = -self.alp[:-1,:-1]*self.ustar[1:-2,:-1]
        t4 = -self.bet[1:,:-1]*self.ustar[2:-1,:-1]
        t5 = -self.phi[1:-1,1:]*self.vstar[1:-1,2:]
        t6 =  self.phi[1:-1,:-1]*self.vstar[1:-1,:-2]

        t7 = -1./self.d*(self.K[1:-1, 1:] + self.h[2:-2, 2:-1] - self.K[1:-1,:-1] - self.h[2:-2,1:-2])

        if self.udrag["type"] == "rayleigh":
            drag = -self.v[2:-2,2:-2]/self.udrag["tau_u"]
        else:
            drag = 0.0
            
        return t1 + t2 + t3 + t4 + t5 + t6 + t7 + drag

    def tendencies(self):
        # Make sure boundary conditions have been applied
        """Refresh boundaries and diagnostics, then return state tendencies.
        
        Returns
        -------
        dict of str to numpy.ndarray
            Time derivatives keyed by ``h``, ``u`` and ``v``, with shapes
            ``(Nx, Ny)``, ``(Nx + 1, Ny)`` and ``(Nx, Ny + 1)`` respectively.
            These match the grid's prognostic slices.
        
        Notes
        -----
        Updates halos and diagnostic arrays in place but does not advance
        the simulation time or apply a time-integration step."""
        self.apply_bcs()
        self.update_diagnostics()
        return {
            "h": self.dh_dt(),
            "u": self.du_dt(),
            "v": self.dv_dt()
        }

    def total_E(self):
        """Sum the kinetic and potential energy over the domain"""

        # Potential energy
        PE = 0.5*self.h

        # Sum over domain (ignoring ghosts)
        self.E = np.sum(self.h[2:-2,2:-2]*(self.K[1:-1,1:-1] + PE[2:-2,2:-2]))

    def total_enstrophy(self):
        """Sum total entstrophy over the domain"""

        # Get hq on edge points
        hq = 0.25*(self.h[1:,1:] + self.h[:-1,1:] + self.h[1:,:-1] + self.h[:-1,:-1])
        
        self.enstrophy = np.sum(0.5*hq[self.realslice['q']]*self.q[self.realslice['q']]**2)
        
    def update_diagnostics(self):
        """Refresh kinetic energy, potential vorticity, fluxes and coefficients.
        
        Updates ``K``, ``q``, ``ustar``, ``vstar`` and the six vorticity
        coefficients in dependency order. Boundary values must already be
        current. Does not refresh total energy or enstrophy. Returns None."""
        self.calc_KE()
        self.calc_vorticity()
        self.calc_ustars()
        self.calc_coeffs()


    def io_diagnostics(self):
    # Variables not required for timestepping
        """Refresh boundaries, derived fields, total energy and enstrophy.
        
        Mutates the grid's halos and diagnostic attributes in place.
        Intended for output and printed diagnostics; returns None."""
        self.apply_bcs()
        self.update_diagnostics()
        self.total_E()
        self.total_enstrophy()
    
