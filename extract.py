from h5py import File
from os import walk, mkdir
import numpy as np

# data/ is input directory for .mat files
for (_,_, filenames) in walk("data"):

    # For filename in filenames 
    for fn in filenames:

        # Generate directory string, guarantee existence of directory
        outdir = "out/" + fn.replace(".mat", "/")
        try:
            mkdir(outdir)
        except:
            pass

        # Use h5py to read data
        with File("data/" + fn, "r") as data:
            
            print("Starting: " + fn)

            # The keys are different measurements. 
            # For example gp_lfp12, str_lfp3, ...

            done = 0
            amount = len(data.keys())

            for key in data.keys():

                # Generate .csv
                np.savetxt(
                    # Output is dir/key.csv
                    outdir + key + ".csv",

                    # data[key] contains metadata. ["values"] for relevant data.
                    np.array(data[key]["values"]),

                    # Value separator. No spaces, actually saves a lot of storage and I/O.
                    delimiter = ",",

                    # Avoid values like "3e-1", prefer "0.3" for portability
                    # This realization cost a lot of I/O, thank you numpy!
                    fmt = "%f"
                )

                done += 1
                print("\tFinished " + str(done) + "/" + str(amount))

            print("Finished: " + fn)