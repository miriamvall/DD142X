from h5py import File
import numpy as np

# Returns the channels contained in a .mat file
# Returned as a dictionary:
#   {channel name : channel values}
def asDict(filename):
    with File(filename, "r") as data:
        return {
            key: np.array(data[key]["values"]).flatten() 
            for key in data
        }

# Helper function
def by_pattern(filename, pattern, epoch_size = None):
    with File(filename, "r") as data:
        out = np.array([
            np.array(data[key]["values"]).flatten()
            for key in data
            if pattern in key
        ])
        if epoch_size == None:
            return out
        else:
            hi = int(out.shape[1] / epoch_size)
            return out[ : , 0 : hi]

# Returns the str_lfp channels contained in a .mat file
# Returned as a numpy array
# Each row represents the values for a channel
# If epoch_size argument is supplied, attempts trim output such that 
# each channel has the largest possible integer n * epoch_size length
def str_lfp(filename, epoch_size = None):
    return by_pattern(filename, "str_lfp", epoch_size)

# Returns the gp_lfp channels contained in a .mat file
# Returned as a numpy array
# Each row represents the values for a channel
# If epoch_size argument is supplied, attempts trim output such that 
# each channel has the largest possible integer n * epoch_size length
def gp_lfp(filename, epoch_size = None):
    return by_pattern(filename, "gp_lfp", epoch_size)