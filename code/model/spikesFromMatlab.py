from h5py import File
from os import walk, mkdir
import numpy as np
import scipy.io as sio


# All .mat in a directory to a directory/csv-tree of values in outDir.           
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

    allToCsv("../_data/matlabData/entr", "../_data/csvData/summary")




test()