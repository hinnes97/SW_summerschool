import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid

def gaussian(grid, amp, width, x0, y0):
    """Gaussian height bump"""
    xx, yy = np.meshgrid(grid.xm, grid.ym, indexing='ij')
    grid.h = 1 + amp*np.exp(-(xx - x0)**2/width**2)*np.exp(-(yy - y0)**2/width**2)

def vortex(grid, amp, width):
    """Geostrophically balanced vortex
    Note this will not be completely balanced wrt the discretisation scheme and
    gradient operators """
    
    gaussian(grid, amp, width, 0., 0.)

    xu, yu = np.meshgrid(grid.xe, grid.ym, indexing='ij')
    grid.u = 2*yu/width**2*amp*np.exp(-(xu**2 + yu**2)/width**2)*grid.Ro

    xv, yv = np.meshgrid(grid.xm, grid.ye, indexing='ij')
    grid.v = -2*xv/width**2*amp*np.exp(-(xv**2 + yv**2)/width**2)*grid.Ro

def plane_gravity_wave(grid, amp, width):
    """Plane wave moving with gravity wave speed (c = 1)"""

    # Make gaussian bump with essentially no y-structure (~infinite width)
    xx, yy = np.meshgrid(grid.xm, grid.ym, indexing='ij')
    #grid.h = 1. + amp*np.exp(-xx**2/width**2)*np.sin(2*np.pi*xx*3/width)
    grid.h = 1 + amp*np.sin(2.*np.pi*xx/width)
    
    # For gravity waves (travelling in +x direction), u = h - 1
    # Careful! u is not on the same grid as h
    u_h = grid.h - 1
    grid.u[1:-1] = (u_h[:-1] + u_h[:-1])/2.
    grid.u[0] = 0.5*(u_h[0] + u_h[-1])

def rossby(grid, amp):

    """Set a sinusoidal height perturbation and geostrophic velocities.
    
    Parameters
    ----------
    grid : ArakawaCGrid
        Grid providing staggered coordinates and the Rossby parameter
        ``Ro``. Its ``h``, ``u`` and ``v`` arrays are replaced in place.
    amp : float
        Nondimensional amplitude of the height perturbation about one.
    
    Notes
    -----
    Uses zonal and meridional wavenumbers equal to pi. Velocities are
    computed on their respective staggered grids. Boundary conditions
    and diagnostic updates are left to the caller. Returns None."""
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

    """Initialize a zonal jet, balanced height and meridional perturbation.
    
    Parameters
    ----------
    grid : ArakawaCGrid
        Grid providing staggered coordinates, ``Ro`` and ``beta``.
        ``Ro`` must be nonzero for the height-gradient calculation.
    amp : float
        Peak zonal velocity of the squared-secant-hyperbolic jet.
    width : float
        Nonzero meridional width scale in nondimensional coordinates.
    
    Notes
    -----
    Replaces ``u`` and ``h`` using the jet profile and a cumulative
    trapezoidal integration of its height gradient. Adds a fixed
    meridional velocity perturbation (amplitude 0.1, width 0.15) to the
    existing ``v`` array. Boundary conditions and diagnostics are left
    to the caller. Returns None."""
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
    
