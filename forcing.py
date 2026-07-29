import numpy as np

def null_forcing(grid):
    """Returns zero"""
    return np.zeros_like(grid.h)

def gill_forcing(grid):
    """Returns forcing similar to Gill 1980"""
    return grid.forcing_params['Q']*np.exp(-grid.lam/2*self.ym)
