import numpy as np

def gaussian(grid, x0, y0, lx, ly, amp):
    """Gaussian height bump"""
    xx, yy = np.meshgrid(grid.xm, grid.ym, indexing='ij')
    grid.h = 1 + amp*np.exp(-(xx - x0)**2/lx**2)*np.exp(-(yy - y0)**2/ly**2)
    
def vortex(grid, r0, amp):
    """Geostrophically balanced vortex
    Note this will not be completely balanced wrt the discretisation scheme and
    gradient operators """
    
    gaussian(grid, 0., 0., r0, r0, amp)

    xu, yu = np.meshgrid(grid.xe, grid.ym, indexing='ij')
    grid.u = 2*yu/r0**2*amp*np.exp(-(xu**2 + yu**2)/r0**2)

    xv, yv = np.meshgrid(grid.xm, grid.ye, indexing='ij')
    grid.v = -2*xv/r0**2*amp*np.exp(-(xv**2 + yv**2)/r0**2)
    
    
    
