from h5py import File
from os import walk, mkdir
import numpy as np

# Can be used to skip to any I/O steps of extracting matlab data.
# If a channel is chosen (e.g. str_lfp1) returns values from that channel as a numpy array. (1D)
# Else, return a dict of <channel: str> : <values: 1D numpy array>
def getMatlabValues(fileName, channel = None):

    with File(fileName, "r") as data:
        if channel != None:
            return np.array(data[channel]["values"]).flatten()
        else:
            values = {
                key: np.array(data[key]["values"]).flatten() for key in data.keys()
            }
            return values

# Returns all keys in a matlab file. Ex [gp_lfp1, gp_lfp2, ... str_lfp1, str_lfp2, ...]
def getMatlabKeys(fileName):
    with File(fileName, "r") as data:
        return data.keys()

# All .mat in a directory to a directory/csv-tree of values in outDir.
# Example:
# Input:    
#   dir/
#       NPR-075.b11.mat
#       NPR-076.b11.mat
# Output:
#       NPR-075.b11/
#           gp_lfp1.csv
#           gp_lfp2.csv
#           ...
#       NPR-076.b11/
#           str_lfp1.csv
#           str_lfp2.csv
#           ...
#               
def allToCsv(inDir, outDir):
    # Ensure valid out directory
    try:
        mkdir(outDir)
    except:
        pass
    for (_,_, filenames) in walk(inDir):
        # For filename in filenames 
        for fn in filenames:
            # Generate directory string, guarantee existence of directory
            nestedOutDir = outDir + "/" + fn.replace(".mat", "/")
            try:
                mkdir(nestedOutDir)
            except: 
                pass
            # Use h5py to read data
            with File(inDir + "/" + fn, "r") as data:
                print("Starting: " + fn)
                # The keys are different measurements. 
                # For example gp_lfp12, str_lfp3, ...
                done = 0
                amount = len(data.keys())
                for key in data.keys():
                    # Generate .csv
                    np.savetxt(
                        # Output is dir/key.csv
                        nestedOutDir + key + ".csv",
                        # data[key] contains metadata. ["values"] for relevant data.
                        np.array(data[key]["values"]),
                        # Value separator. No spaces, actually saves a lot of storage and I/O.
                        delimiter = ",",
                        # Avoid values like "3e-1", prefer "0.3" for portability
                        fmt = "%f"
                    )
                    done += 1
                    print("\tFinished " + str(done) + "/" + str(amount))
                print("Finished: " + fn)

allToCsv("matlabData", "csvData")