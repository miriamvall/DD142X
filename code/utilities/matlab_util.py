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

# Checks for the presence of a pattern in the set of keys for a .mat file
def has_pattern(filename, pattern):
    with File(filename, "r") as data:
        keys = data.keys()
        for key in keys:
            if pattern in key:
                return True
        return False

def has_str(filename):
    return has_pattern(filename, "str_lfp")

def has_gp(filename):
    return has_pattern(filename, "gp_lfp")

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
            return out[ : , 0 : hi * epoch_size]

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