import numpy as np

class ArakawaCGrid:

    def __init__(self, N, Ro, bc='periodic', forcing = "None"):
        """Creates Arakawa C-grid object with 2*N x points and N y-points"""

        # Set resolution
        self.d = 1/N

        # Make equatorial beta plane 2 times longer than tall (planetary scale)
        self.Nx = 2*N
        self.Ny = N

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

        # Set up grids, including halo points
        # Main variables u,v,h
        self.u = np.zeros((self.Nx+5, self.Ny+4))
        self.v = np.zeros((self.Nx+4, self.Ny+5))
        self.h = np.zeros((self.Nx+4, self.Ny+4))

        # Gravity parameter
        #self.g = g
        # Planetary scale waves - use Rossby number
        self.Ro = Ro
        
        # Helper variables
        self.ustar = np.zeros((self.Nx+3, self.Ny+2)) # mass flux on u points
        self.vstar = np.zeros((self.Nx+2, self.Ny+3)) # mass flux on v points

        # Potential vorticity and kinetic energy
        self.q = np.zeros((self.Nx+3, self.Ny+3))
        self.K = np.zeros((self.Nx+2, self.Ny+2))

        # Coefficient grids (eq 3.34 Arakawa and Lamb)
        self.eps = np.zeros((self.Nx+2, self.Ny+2))
        self.phi = np.zeros((self.Nx+2, self.Ny+2))
        self.alp = np.zeros((self.Nx+1, self.Ny+2))
        self.bet = np.zeros((self.Nx+1, self.Ny+2))
        self.gam = np.zeros((self.Nx+1, self.Ny+2))
        self.det = np.zeros((self.Nx+1, self.Ny+2))

        # Set y boundary condition
        self.ybc = bc

        # Set forcing
        if forcing not None:
            if forcing["type"] = "Gaussian":
                self.forcing = force.gaussian(self)
        else:
            self.forcing = np.zeros_like(self.h)
                
    def __iter__(self):
        yield "u", self.u
        yield "v", self.v
        yield "h", self.h
        
    def calc_coeffs(self):
        """Updates coefficients on the Arakawa C-Grid"""

        # These are defined on the h grid, inc. ghost cells
        self.eps = 1./24. * (q[1:,1:] + q[:-1, 1:]  - q[:-1, :-1] - q[1:,:-1])
        self.phi = 1./24. *  (-q[1:,1:] + q[:-1,1:]  + q[:-1,:-1] - q[1:,:-1])

        # All defined at u points (i,j+1/2), including y ghosts but not with x ghost
        self.alp = 1./24. * (2*q[2:,1:] + q[1:-1, 1:] + 2*q[1:-1,:-1] + q[2:,:-1])
        self.bet = 1./24. * (q[1:-1,1:] + 2*q[:-2,1:] + q[:-2,:-1] + 2*q[1:-1,:-1])
        self.gam = 1./24. * (2*q[1:-1,1:] + q[:-2,1:] + 2*q[:-2,:-1] + q[1:-1,:-1])
        self.det = 1./24 *  (q[2:,1:] + 2*q[1:-1, 1:] + q[1:-1,:-1] + 2*q[2:,:-1])

    def calc_ustars(self):
        hu = 0.5*(self.h[:-1,:] + self.h[1:,:])
        hv = 0.5*(self.h[:,:-1] + self.h[:,1:])

        self.ustar[1:-1,:] = hu*self.u[1:-1,:]
        self.vstar[:,1:-1] = hv*self.v[:,1:-1]

        # Apply boundary conditions
        [self.ustar, self.vstar] = self.apply_bc([ustar, vstar])

    def apply_bcs(self, var):

        for v in var:
            v[0, :] = v[-4,:]
            v[1, :] = v[-3,:]
            v[-1,:] = v[3,:]
            v[-2,:] = v[2,:]

        if self.ybc == 'free-slip-wall':
            self.v[:,1] = 0.
            self.v[:,-2] = 0.

            for v in var:
                v[:,0] = v[:,2]
                v[:,1] = v[:,2]
                v[:,-1] = v[:,-3]
                v[:,-2] = v[:,-3]

        elif "periodic":
            for v in var:
                v[:,0] = v[:,-4]
                v[:,1] = v[:,-3]
                v[:,-1] = v[:,3]
                v[:,-2] = v[:,2]
            
        return var


    def coriolis(self):
        # Only equatorial beta-plane currently implemented, in non-dimensionalised form
        y = np.linspace(-0.5, 0.5, self.Ny+1)
        return self.Ro*y
        
    def calc_vorticity(self):
        """Calculate the potential vorticity q = (zeta + f)/h
           Definition of variables Arakawa and Lamb eq. 3.11, 3.12"""

        # Relative vorticity defined on the q points
        zeta = 1./self.d * (self.u[1:-1,:-1] - self.u[1:-1,1:] + self.v[1:,1:-1] - self.v[:-1,1:-1])

        # Require h on q grid
        hq = 1./4.*(h[1:,1:] + h[:-1,1:] + h[1:,:-1] + h[:-1,:-1])

        f = self.coriolis()

        self.q[1:-1,1:-1] = (zeta + f)/hq
        [self.q] = self.apply_bc([self.q])

    def calc_KE(self):
        """Kinetic energy at h grid (centres)"""
        
        usq = 0.5*((self.u**2)[0:-1,:] + (self.u**2)[1:,:])
        vsq = 0.5*((self.v**2)[:,0:-1] + (self.v**2)[:,1:])

        self.K = 0.5*(usq + vsq)


    def divergence(self):
        """"Calculating the divergence of the mass flux (Arakawa and Lamb, eq. 3.2)
            Do not bother calculating ghost cells, only needed to update h"""

        return 1./self.d * (self.ustar[2:-1,1:-1] - self.ustar[1:-2,1:-1] + self.vstar[1:-1,2:-1] - self.vstar[1:-1,1:-2])

    def dh_dt(self):
        """Return dh_dt from equation 3.1, AL80"""
        return -self.divergence() + self.forcing


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

        #phi = self.g*self.h
        t7 = -1./self.d *(self.K[1:,1:-1] + self.h[1:,1:-1] - self.K[:-1,1:-1] - self.h[:-1,1:-1])

        return t1 + t2 + t3 + t4 + t5 + t6 + t7

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

        t7 = -1./self.d*(self.K[1:-1, 1:] + self.h[1:-1, 1:] - self.K[1:-1,:-1] - self.h[1:-1,:-1])

        return t1 + t2 + t3 + t4 + t5 + t6 + t7

    def tendencies(self):
        # Make sure boundary conditions have been applied
        [self.h, self.u, self.v] = self.apply_bcs([self.h, self.u, self.v])
        
        # Update kinetic energy, vorticity and coeffs
        self.diagnostics()
        
        return {
            "h": self.dh_dt(),
            "u": self.du_dt(),
            "v": self.dv_dt()
        }

    def diagnostics(self):
        self.calc_KE()
        self.calc_vorticity()
        self.calc_ustars()
        self.calc_coefficients()
        
        
