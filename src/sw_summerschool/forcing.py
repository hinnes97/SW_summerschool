import numpy as np

def null_forcing(grid):
    """Returns zero"""
    return np.zeros_like(grid.h[2:-2,2:-2])

def exoplanet_forcing(grid):
    """Exoplanet-like stationary forcing. Apply to cell on non ghost grid"""
    xm, ym = np.meshgrid(grid.xm, grid.ym, indexing='ij')

    # Choose arbitrary forcing amplitude (probably should be in config file)
    A = 0.1
    # Note we need to divide by tau_h (force damping timescale ) to be consistent
    return (1. + A*np.cos(np.pi*xm[2:-2,2:-2])*np.cos(np.pi*ym[2:-2,2:-2]))/grid.hdrag["tau_h"]
