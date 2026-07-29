import numpy as np
import yaml
from funcs import ArakawaCGrid
from adam_bashford import AB3
from inout import IO

class SW_model:
    def __init__(self, cfg_file):

        # Read in config
        with open(cfg_file, 'r') as f:
            cfg = yaml.safe_load(f)

        # Create grid object
        self.grid = ArakawaCGrid(cfg["N"], cfg["lam"], cfg["bc"], cfg["forcing"],
                                 cfg["ic"], cfg["udrag"], cfg["hdrag"], cfg['cor'])

        # Create integrator
        self.tstep = cfg["tstep"]
        self.tstepper = AB3(self, self.tstep)
        self.time = 0.0

        # Input-output
        self.io = IO(cfg['outfile'], cfg['outvars'])
        self.io.initialise_output(self.grid)
        
        # Do first save
        self.io.write_variables(self.time, self.grid)

        print("-------------- Initialised ----------------------")
        print("Total Energy: ", self.grid.E)
        print("Total Enstrophy: ", self.grid.enstrophy)
        print("--------------------------------------------------")
            
    def integrate(self, Nt):
        # Integrate the shallow water model forward Nt times
        for i in range(1,Nt+1,1):
            self.tstepper.step()
            
            if i%cfg["io_freq"]==0:
                self.io.write_variables(self.time, self.grid)

            if i%cfg["print_freq"]==0:
                print("-----------------------------------------------")
                print("Time: ", self.time)
                print("Total Energy: ", self.grid.E)
                print("Total Enstrophy: ", self.grid.enstrophy)
            
            

