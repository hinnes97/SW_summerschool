import numpy as np

def null_forcing(grid):
    """Returns zero"""
    return np.zeros_like(grid.h[2:-2,2:-2])

def exoplanet_forcing(grid):
    raise NotImplementedError()
