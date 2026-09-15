import netCDF4
import os

class IO:
    def __init__(self, outfile, outvars):
        """Store the destination and variable selection for NetCDF output.
        
        Parameters
        ----------
        outfile : str or path-like
            Destination file, relative to the working directory if needed.
        outvars : sequence of str
            Grid variable names to save; each must have output dimensions
            and a real-domain slice defined by the grid.
        
        Notes
        -----
        No file is opened until ``initialise_output`` is called."""
        self.outfile = outfile
        self.outvars = outvars


    def initialise_output(self, grid):
        """Initialise output on the uniform centred grid"""
        if os.path.exists(self.outfile):
            os.remove(self.outfile)
            
        self.data = netCDF4.Dataset(self.outfile, "w", format="NETCDF4")

        # Initialise dimensions
        self.time = self.data.createDimension("time", None)
        self.ymid = self.data.createDimension("ymid", grid.Ny)
        self.xmid = self.data.createDimension("xmid", grid.Nx)

        if grid.ybc =='free-slip-wall':
            self.yedge = self.data.createDimension("yedge", grid.Ny+1)
        elif grid.ybc=='periodic':
            self.yedge = self.data.createDimension("yedge", grid.Ny)
            
        self.xedge = self.data.createDimension("xedge", grid.Nx)

        # initialise output variables
        self.times = self.data.createVariable("time", "f8", ("time",))
        self.xmids = self.data.createVariable("xmid", "f8", ("xmid",))
        self.ymids = self.data.createVariable("ymid", "f8", ("ymid",))
        self.xedges = self.data.createVariable("xedge", "f8", ("xedge",))
        self.yedges = self.data.createVariable("yedge", "f8", ("yedge",))
        
        for var in self.outvars:
            self.data.createVariable(var, "f8", grid.vargrid[var])

        # Write grid variables
        self.xmids[:] = grid.xm[2:-2]
        self.ymids[:] = grid.ym[2:-2]
        self.xedges[:] = grid.xe[2:-3] # Always periodic

        if grid.ybc == "free-slip-wall":
            self.yedges[:] = grid.ye[2:-2]
        elif grid.ybc == 'periodic':
            self.yedges[:] = grid.ye[2:-3]

        # Make boundary condition attribute
        self.data.ybc = grid.ybc
        
    def write_variables(self, time, grid):
        """Append one time record and the selected grid variables.
        
        Parameters
        ----------
        time : float
            Nondimensional simulation time associated with this record.
        grid : ArakawaCGrid
            Source state. Variables are indexed by their real-domain slices
            to omit halo points.
        
        Notes
        -----
        Requires an output file opened by ``initialise_output``. Refreshes
        IO diagnostics if energy or enstrophy is requested, which also
        updates the grid's boundary values and diagnostic arrays. Returns
        None and leaves the file open."""
        it = len(self.times)
        self.times[it] = time

        if "E" in self.outvars or "enstrophy" in self.outvars:
            grid.io_diagnostics()
            
        for var in self.outvars:
            self.data[var][it] = grid[var][grid.realslice[var]]

    def close(self):
        """Close the initialized NetCDF dataset and flush pending writes.
        
        Returns None. Call after output initialization and after the last
        write; the closed dataset cannot accept further records."""
        self.data.close()

