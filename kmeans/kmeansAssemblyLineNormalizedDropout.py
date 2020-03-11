import pandas as pd 
import numpy as np 
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from mpl_toolkits.mplot3d import Axes3D
from os import walk
from math import floor
from random import sample

#####

# Normalize the data before running old tests
# Also use dropout

#####

inputX = 193                    # Currently time points
inputY = 28                     # Frequencies per file
chunks = 35                     # 100s/chunks length chunks
Xchunkd = floor(193/chunks)     # Amount of time points per chunk
inputN = 27                     # Amount of files
Nclusters = 8                   # Number of clusters

fracForFit = 0.5                # Fraction of samples used for k-Means fit

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
            # i = range 0:(inputN-1), j = range 0:(chunks-1)
            # Read each chunk, reshape to be vector instead of matrix

            # Normalization:    x_i_normalized = (x_i - x_mean)/x_stdiv
            preNormalized = arr[:, j * Xchunkd   :   (j + 1) * Xchunkd].reshape((inputY * Xchunkd, ))
            postNormalized = (preNormalized - preNormalized.mean()) / preNormalized.std()
            data[i * chunks + j] = postNormalized
            
# Generate k-Means model
kmeans = KMeans(n_clusters = Nclusters).fit(
    np.array(
        # Only take sample of rows
        sample([row for row in data], floor(fracForFit * data.shape[0]))
    )
)
# Predictions
predictions = kmeans.predict(data).reshape((inputN, chunks))
# Amount of equal-prediction columns
allSame = np.array(
    # If all values in column i are the same, they are the same as the first at [0, i]
    # The == returns an array of values True or False
    # all returns true if all values are true
    # Convert to int (true -> 1) and sum
    [all(predictions[0, i] == predictions[:,i]) for i in range(0, predictions.shape[1])], 
    dtype=int).sum()
# Centers
clusters = kmeans.cluster_centers_
# Distance matrix of centers
distanceMatrix = np.array([np.array([np.linalg.norm(x - y) for x in clusters]) for y in clusters])
# To output means of all clusters (centers)
clusterMeans = np.zeros((Nclusters, Xchunkd * inputY))
# To output standard deviation of cluster center means
clusterStds = np.zeros((Nclusters, Xchunkd * inputY))
# Helper dictionary
tmpDir = {}

for i in range (0, Nclusters):
    # Will contain vectors predicted to a certain cluster
    tmpDir[i] = []

# Create lists of same-cluster-predicted vectors
for (x, c) in zip(data, kmeans.predict(data)):
    for i in range(0, Nclusters):
        if (c == i):
            tmpDir[i].append(x)

# Calculate means, standard deviation
for i in range(0, Nclusters):
    clusterMeans[i] = np.array(tmpDir[i]).mean(axis = 0)
    clusterStds[i] = np.array(tmpDir[i]).std(axis = 0)

# Output data
files = [
    ["clusterMeans", clusterMeans],
    ["clusterStds", clusterStds],
    ["clusterMeansStds", clusterMeans.std(axis = 0)],
    ["predictions", predictions],
    ["centers", kmeans.cluster_centers_],
    ["clusterDistances", distanceMatrix],
    ["amountCorrect", np.array((allSame, predictions.shape[1]))]
]

for (fn, xs) in files:
    np.savetxt("outNormalizedDropout/" + fn + ".csv", xs, delimiter = ",")