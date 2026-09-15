import numpy as np
import yaml
from .arakawa_c import ArakawaCGrid
from .adam_bashford import AB3
from .inout import IO

class SW_model:
    def __init__(self, cfg_file, **config):

        # Read in config
        """Initialize the model from YAML and write its initial state.
        
        Parameters
        ----------
        cfg_file : str or path-like
            YAML file containing grid, forcing, initial-condition, drag,
            time-step and output settings. Relative paths use the current
            working directory.
        **config
            Top-level configuration overrides; nested mappings are replaced,
            not merged.
        
        Notes
        -----
        Creates the grid and AB3 integrator, resets time to zero, opens the
        configured NetCDF output (replacing any existing file), writes the
        initial state and prints initial diagnostics. Close ``self.io`` when
        finished to release the output file."""
        with open(cfg_file, 'r') as f:
            cfg = yaml.safe_load(f)

        # Override config with manual entries
        for c in config:
            cfg[c] = config[c]
            
        self.cfg = cfg
        # Create grid object
        self.grid = ArakawaCGrid(cfg["N"], cfg["Ro"], cfg["bc"], cfg["forcing"],
                                 cfg["ic"], cfg["udrag"], cfg["hdrag"], 
                                 beta = cfg["beta"])

        # Create integrator
        self.tstep = cfg["tstep"]
        self.tstepper = AB3(self, self.tstep)
        self.time = 0.0
        self.Nt_tot = 0

        # Input-output
        self.io = IO(cfg['outfile'], cfg['outvars'])
        self.io.initialise_output(self.grid)
        
        # Do first save
        self.io.write_variables(self.time, self.grid)
        self.Nt_tot += 1
        
        print("-------------- Initialised ----------------------")
        print("Total Energy: ", self.grid.E)
        print("Total Enstrophy: ", self.grid.enstrophy)
        print("--------------------------------------------------")
            
    def integrate(self, Nt):
        # Integrate the shallow water model forward Nt times
        """Advance the model in place by ``Nt`` additional time steps.
        
        Parameters
        ----------
        Nt : int
            Number of steps to take with the configured time-step size.
        
        Returns
        -------
        None
            Updates the grid, simulation time and cumulative step counter.
            Writes output every ``io_freq`` steps and prints diagnostics
            every ``print_freq`` steps. The output file remains open."""
        for i in range(self.Nt_tot,self.Nt_tot + Nt):
            self.tstepper.step()
            if i%self.cfg["io_freq"]==0:
                print(i, self.cfg["io_freq"])
                self.io.write_variables(self.time, self.grid)

            if i%self.cfg["print_freq"]==0:
                self.grid.io_diagnostics()
                print("-----------------------------------------------")
                print("Time: ", self.time)
                print("Total Energy: ", self.grid.E)
                print("Total Enstrophy: ", self.grid.enstrophy)
            
            self.Nt_tot += 1

