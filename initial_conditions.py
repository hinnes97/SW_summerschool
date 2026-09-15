import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid

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
    grid.u = 2*yu/r0**2*amp*np.exp(-(xu**2 + yu**2)/r0**2)*grid.Ro

    xv, yv = np.meshgrid(grid.xm, grid.ye, indexing='ij')
    grid.v = -2*xv/r0**2*amp*np.exp(-(xv**2 + yv**2)/r0**2)*grid.Ro

def plane_gravity_wave(grid, amp, width):
    """Plane wave moving with gravity wave speed (c = 1)"""

    # Make gaussian bump with essentially no y-structure (~infinite width)
    xx, yy = np.meshgrid(grid.xm, grid.ym, indexing='ij')
    grid.h = 1. + amp*np.exp(-xx**2/width**2)*np.sin(2*np.pi*xx*3/width)
    #grid.h = 1 + amp*np.sin(2.*np.pi*xx/width)
    
    # For gravity waves (travelling in +x direction), u = h - 1
    # Careful! u is not on the same grid as h
    u_h = grid.h - 1
    grid.u[1:-1] = (u_h[:-1] + u_h[:-1])/2.
    grid.u[0] = 0.5*(u_h[0] + u_h[-1])

def rossby(grid, amp):

    k = np.pi
    l = np.pi

    # h grid
    xh, yh = np.meshgrid(grid.xm, grid.ym, indexing="ij")

    # u grid
    xu, yu = np.meshgrid(grid.xe, grid.ym, indexing="ij")

    # v grid
    xv, yv = np.meshgrid(grid.xm, grid.ye, indexing="ij")

    # Height perturbation
    grid.h = (
        1.0
        + amp
        * np.cos(k*xh)
        * np.sin(l*(yh + 0.5))
    )

    # Geostrophic u
    grid.u = (
        -amp * l * grid.Ro
        * np.cos(k*xu)
        * np.cos(l*(yu + 0.5))
    )

    # Geostrophic v
    grid.v = (
        -amp * k * grid.Ro
        * np.sin(k*xv)
        * np.sin(l*(yv + 0.5))
    )

def barotropic_jet(grid, amp, width):

    xx, yy = np.meshgrid(grid.xm, grid.ym, indexing='ij')
    xe, ym = np.meshgrid(grid.xe, grid.ym, indexing='ij')
    
    grid.u = amp/np.cosh(ym/width)**2

    dhdy = -1./grid.Ro*(1. + grid.beta*grid.ym)*amp*1./np.cosh(grid.ym/width)**2
    h_raw = cumulative_trapezoid(dhdy, grid.ym, initial=0.0)

    C = 1 - trapezoid(h_raw[2:-2], grid.ym[2:-2])

    h = h_raw + C
    grid.h = np.ones_like(xx)*h[None,:]

    # V perturbation
    kx = 4*np.pi / 2.0   # one wave across x-domain [-1,1]
    xv, yv = np.meshgrid(grid.xm, grid.ye, indexing="ij")
    eps = 1.e-1
    Lp = 0.15
    grid.v += (
        eps
        * np.cos(kx*xv)
        * np.exp(-(yv/Lp)**2)
    )
    
