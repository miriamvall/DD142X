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
Nclusters = 8                   # Number of clusters

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
            
kmeans = KMeans(n_clusters = Nclusters).fit(data)
clusters = kmeans.cluster_centers_
distanceMatrix = np.array([np.array([np.linalg.norm(x - y) for x in clusters]) for y in clusters])

clusterMeans = np.zeros((Nclusters, Xchunkd * inputY))
clusterStds = np.zeros((Nclusters, Xchunkd * inputY))
tmpDir = {}
for i in range (0, Nclusters):
    tmpDir[i] = []
for (x, c) in zip(data, kmeans.predict(data)):
    for i in range(0, Nclusters):
        if (c == i):
            tmpDir[i].append(x)
for i in range(0, Nclusters):
    clusterMeans[i] = np.array(tmpDir[i]).mean(axis = 0)
    clusterStds[i] = np.array(tmpDir[i]).std(axis = 0)

files = [
    ["clusterMeans", clusterMeans],
    ["clusterStds", clusterStds],
    ["clusterMeansStds", clusterMeans.std(axis = 0)],
    ["kMeans", kmeans.predict(data).reshape((inputN, chunks))],
    ["centers", kmeans.cluster_centers_],
    ["clusterDistances", distanceMatrix]
]

for (fn, xs) in files:
    np.savetxt("out/" + fn + ".csv", xs, delimiter = ",")