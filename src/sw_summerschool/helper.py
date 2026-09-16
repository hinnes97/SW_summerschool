import numpy as np

def interp_to_centre(var, ds, bc=None):
    """Useful function for interpolating to single output grid"""
    data = ds[var].data
    if var=="q":
        # On grid cell corners, must be interpolated to centres
        if bc=='periodic':
            # Roll both x and y
            return 0.25*(data + np.roll(data, 1, axis=1) + np.roll(data,1,axis=2) + np.roll(np.roll(data,1,axis=1),1,axis=2))
        else:
           # Only roll y
            return 0.25*(data[:,:,:-1] + np.roll(data, 1, axis=1)[:,:,:-1] + data[:,:,1:] + np.roll(data, 1, axis=1)[:,:,1:])
    elif var=="u":
        return 0.5*(data + np.roll(data, 1, axis=1))
    elif var=="v":
        if bc=="periodic":
            # Roll
            return 0.5*(data + np.roll(data, 1, axis=2))
        else:
            return 0.5*(data[:,:,:-1] + data[:,:,1:])
