import numpy as np

def null_forcing(grid):
    """Returns zero"""
    return np.zeros_like(grid.h[2:-2,2:-2])

def gill_forcing(grid):
    """Returns forcing similar to Gill 1980"""

    cos = np.cos(np.pi*grid.xm[:,None])
    cos = np.where(cos<0, 0., cos)
    
    q = 1. + grid.forcing_params["Q"]*np.exp(-grid.lam/2*grid.ym[None,:]**2)*cos
    
    force =  q/grid.hdrag["tau_h"]
    force = force[grid.realslice["h"]]
    return force
