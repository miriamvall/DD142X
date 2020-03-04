import pandas as pd 
import numpy as np 
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from mpl_toolkits.mplot3d import Axes3D
from os import walk
from math import floor

inputX = 193                    # Currently time points
inputY = 28                     # Frequencies per file
chunks = 35                     # 100s/chunks => 10s per chunk
Xchunkd = floor(193/chunks)     # Amount of time points per chunk
inputN = 27                     # Amount of files

# Shape: (files * chunks/file) * (frequencies * N time measurments)
data = np.zeros((inputN * chunks, inputY * Xchunkd))

# Input data
for (dirpath, dirnames, filenames) in walk("../fourierdata/"):

    # For each filename, and it's "index"
    for (i, fn) in enumerate(sorted(filenames)):
        # Read the data, limit it to chunks * length of a chunk length
        arr = np.genfromtxt(dirpath + fn, delimiter = ",")[:,0:chunks * Xchunkd]
        # For the amount of chunks you want...
        for j in range(0, chunks):
            # i = 0:(inputN-1), j = 0:(chunks-1)
            # Read each chunk, reshape to be vector instead of matrix
            data[i * chunks + j] = arr[:, j * Xchunkd   :   (j + 1) * Xchunkd].reshape((inputY * Xchunkd, ))

kmeans = KMeans(n_clusters = 8).fit(data)
#print(kmeans.predict(data).reshape((inputN, chunks)))
#np.savetxt("kMeans.csv", kmeans.cluster_centers_[0] - kmeans.cluster_centers_[1], delimiter = ",")
np.savetxt("kMeans.csv", kmeans.predict(data).reshape((inputN, chunks)).astype(int), delimiter = ",")
np.savetxt("centers.csv", kmeans.cluster_centers_, delimiter = ",")