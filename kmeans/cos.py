import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import pairwise_distances_argmin_min
from mpl_toolkits.mplot3d import Axes3D
from os import walk
import csv
from scipy import spatial

# STORING (and reshaping) THE 27 SAMPLES - initially, each of size 28x193 (28*193 = 5404)

data = np.zeros((27, 5404)) # all the samples together

# Input data
for (dirpath, dirnames, filenames) in walk("../fourierdata/"):

    # For each file
    for (i, fn) in enumerate(sorted(filenames)):
        # Read the data
        arr = np.genfromtxt(dirpath + fn, delimiter = ",")[:,:]
        #reshape to be vector instead of matrix
        data[i] = arr[:, :].reshape((5404))

#cosine similarity matrix
cos_matrix = np.zeros((27,27))

#spatial.distance.cosine computes theance, must substract the value
#from 1 to get the similarity

for i in range(0,27):
	for j in range(0,27):
		setI = data[i]
		setII = data[j]
		cos_sim = 1 - spatial.distance.cosine(setI,setII)
		cos_matrix[i,j] = cos_sim

np.savetxt("cos_sim.csv", cos_matrix, delimiter = ",")
