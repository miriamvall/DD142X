import numpy as np
from math import floor

dir = "out/NPR-075.b13/"
files = [
    "gp_lfp1" , "gp_lfp2" , "gp_lfp3" , 
    "gp_lfp4" , "gp_lfp5" , "gp_lfp6" , 
    "gp_lfp7" , "gp_lfp8" , "gp_lfp9" , 
    "gp_lfp17", "gp_lfp18", "gp_lfp20", 
    "gp_lfp21", "gp_lfp22", "gp_lfp23",
    "str_lfp11", "str_lfp12", "str_lfp13", 
    "str_lfp14", "str_lfp15", "str_lfp16", 
    "str_lfp26", "str_lfp27", "str_lfp28", 
    "str_lfp29", "str_lfp30", "str_lfp31"
]

width = 2 ** 13

for fn in files:
    print("Starting " + fn)
    vals = np.genfromtxt(dir + fn + ".csv", delimiter = ",")
    epochs = floor(vals.shape[0] / width)
    output = np.zeros((width, epochs))
    for i in range(0, epochs):
        epochValues = vals[i * width : (i + 1) * width]
        toSpect = np.fft.fft(epochValues, width)
        toAmp = np.abs(toSpect)
        output[:,i] = toAmp
    np.savetxt("fourierdata/" + fn + ".csv", output, delimiter=",")
    print("Finished " + fn)