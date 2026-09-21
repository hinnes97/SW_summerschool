import numpy as np

def null_forcing(grid):
    """Returns zero"""
    return np.zeros_like(grid.h[2:-2,2:-2])

def exoplanet_forcing(grid):
    cos = np.cos(np.pi*grid.xm[:,None])*np.cos(np.pi*grid.ym[None,:])
    q = 1. + grid.forcing_params["amp"]*cos
    force =  q/grid.hdrag["tau_h"]
    force = force[grid.realslice["h"]]
    
    return force
