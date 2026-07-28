import numpy as np
import yaml
from funcs import ArakawaCGrid
from adam_bashford import AB3

class SW_model:
    def __init__(self, cfg_file):

        # Read in config
        with open(cfg_file, 'r') as f:
            cfg = yaml.load(f, load=yaml.FullLoader)

        # Create grid object
        self.grid = ArakawaCGrid(cfg["N"], cfg["Ro"], cfg["bc"], cfg["forcing"])

        # Create integrator
        self.tstepper = AB3(self, cfg["tstep"])
        self.time = 0.0

    def integrate(self, Nt):

        # Integrate the shallow water model forward Nt times
        for i in range(Nt):
            self.grid.calc_KE()

