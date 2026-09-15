import numpy as np
from scipy.integrate import RK45
from collections import deque

class AB3:
    def __init__(self, model, tstep):

        # Timestep
        """Create an AB3 integrator with an initially empty tendency history.
        
        Parameters
        ----------
        model : SW_model
            Model whose grid is updated and whose time is advanced.
        tstep : float
            Retained in the calling interface; the implementation takes its
            step size from ``model.tstep`` instead of this argument.
        
        Notes
        -----
        The grid is shared with the model, not copied. The history retains
        up to three tendency dictionaries for subsequent AB3 steps."""
        
        self.tstep = model.tstep
        self.model = model
        # Define the three states required for timestepping
        self.grid = model.grid
        self.history = deque(maxlen=3)

        self.c0 = 23./12.
        self.c1 = -16./12.
        self.c2 = 5./12.

        self.coeffs = [self.c0, self.c1, self.c2]

    def step(self):

        """Advance the shared grid and model time by one time step.
        
        Uses RK4 bootstrapping while fewer than three tendency records are
        available, then applies the three-step Adams-Bashforth formula.
        Updates prognostic slices and tendency history in place. Returns
        None; output writing is handled by the model driver."""
        
        if len(self.history) < 3:
            self.bootstrap()
            return
            
        # Get the new set of tendencies
        tend = self.grid.tendencies()
        # Update the history array
        self.history.appendleft(tend)

        # March variables forward
        for v,var in self.grid:
            var[self.grid.progslice[v]] = var[self.grid.progslice[v]] + self.tstep*(self.c0*self.history[0][v] + self.c1*self.history[1][v] + self.c2*self.history[2][v])

        self.model.time += self.tstep
        #self.grid.update_diagnostics()
        
    def bootstrap(self):
        """Bootstrap RK4 method, to start integration"""

        # Save initial state
        y0 = {"h": self.grid.h.copy(),
              "v": self.grid.v.copy(),
              "u": self.grid.u.copy()}

        k1 = self.grid.tendencies()
        self.history.appendleft(k1)
        
        for v, var in self.grid:
            var[self.grid.progslice[v]] =  y0[v][self.grid.progslice[v]] + 0.5*k1[v]*self.tstep

        #self.grid.update_diagnostics()
        k2 = self.grid.tendencies()

        for v, var in self.grid:
            var[self.grid.progslice[v]] = y0[v][self.grid.progslice[v]] + 0.5*k2[v]*self.tstep

        #self.grid.update_diagnostics()
        k3 = self.grid.tendencies()

        for v, var in self.grid:
            var[self.grid.progslice[v]] = y0[v][self.grid.progslice[v]] + k3[v]*self.tstep

        #self.grid.update_diagnostics()
        k4 = self.grid.tendencies()

        # Now do step
        for v, var in self.grid:
            var[self.grid.progslice[v]] = y0[v][self.grid.progslice[v]] + self.tstep/6. * (k1[v] + 2*k2[v] + 2*k3[v] + k4[v])

        #self.grid.update_diagnostics()
        self.model.time += self.tstep
            
    
        
            
