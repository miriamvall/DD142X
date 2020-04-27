from h5py import File
from os import walk, mkdir
import numpy as np
import scipy.io as sio

# Can be used to skip to any I/O steps of extracting matlab data.
# If a channel is chosen (e.g. str_lfp1) returns values from that channel as a numpy array. (1D)
# channel can also be a list of patterns to filter on
#   Example: ["str_lfp", "gp_lfp"] will give channels "str_lfp1", "str_lfp2", ... "gp_lfp1", ...
#   In this scenario the return value is a 2D numpy array
# Else, return a dict of <channel: str> : <values: 1D numpy array>
# Also returns a documentation wtring for README
def getMatlabValues(fileName, channel = None):

    docString = "### Matlab Data ###" + \
        "\nFILE " + fileName + \
        "\nCHANNEL " + str(channel) + "\n"

    with File(fileName, "r") as data:
        if isinstance(channel, str):
            return np.array(data[channel]["values"]).flatten(), docString
        else:
            values = {
                key: np.array(data[key]["values"]).flatten() for key in data.keys()
            }
            if isinstance(channel, list):
                out = []
                for pattern in channel:
                    out += [v for (k, v) in values.items() if pattern in k]
                return np.array(out), docString

            return values, docString

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
            # Use scipy to read data 
            data = sio.loadmat(inDir + "/" + fn)
            print("Starting: " + fn)
            # The keys are different measurements. 
            # For example gp_lfp12, str_lfp3, ...
            done = 0
            amount = len(data.keys())
            #print("keys: " + str(amount))
            #print(data.keys())
            for key in data.keys():
                 # Generate .csv
                #print(key)
                #print(data[key])
                if(key == "sp_count_gp_sua") or (key == "sp_count_str_sua") or (key == "sp_count_stn_sua") \
                or (key == "spect_ent_gp") or (key == "spect_ent_str") or (key == "spect_ent_stn"):
                    print(key)
                    print(data[key])
                    np.savetxt(
                        # Output is dir/key.csv
                        nestedOutDir + key + ".csv",
                        # data[key] contains metadata. ["values"] for relevant data.
                        np.array(data[key]),
                         # Value separator. No spaces, actually saves a lot of storage and I/O.
                        delimiter = ",",
                        # Avoid values like "3e-1", prefer "0.3" for portability
                        fmt = "%f"
                    )
                done += 1
                print("\tFinished " + str(done) + "/" + str(amount))
            print("Finished: " + fn)

def test():
    # As dictionary
    #arr1 = getMatlabValues("../_data/matlabData/entr/Summary_NPR-075.b11.mat")
    # Specific channel
    #print(arr1["sp_count_gp_sua"])

    # Specific channel
    #arr2 = getMatlabValues("../_data/matlabData/entr/Summary_NPR-075.b11.mat", "gp_lfp1")
    #print(arr2)

    # Will have gp_lfp1, gp_lfp17, gp_lfp18
    #arr3 = getMatlabValues("../_data/matlabData/entr/Summary_NPR-075.b11.mat", ["gp_lfp1"])
    #print(arr3[0])

    # Use channel = ["gp_lfp", "str_lfp"] for relevant data

    allToCsv("../_data/matlabData/entr", "../_data/csvData/summary")




test()