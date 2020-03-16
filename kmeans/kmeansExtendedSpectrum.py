import numpy as np
from os import walk
from sklearn.cluster import KMeans
from random import sample
from math import floor

dir = "../extract/fourierdataSpecific/"

# Bands
ranges = [
    (4, 8),
    (8, 12),
    (12, 30),
    (30, 60),
    (60, 120)
]
nBands = len(ranges)
offset = 4  # For indexing
nClusters = 16
fitFrac = 0.5

data = {}
for (_, _, filenames) in walk(dir):
    for fn in filenames:
        freqOverTime = np.genfromtxt(dir + fn, delimiter = ",")
        # Set each time but stay the same ->
        nFreq = freqOverTime.shape[0]
        nEpochs = freqOverTime.shape[1]
        # <- For convenience
        data[fn] = np.array(    # nBands * nEpochs shape
            [freqOverTime[lo - offset : hi - offset].sum(axis = 0) / (hi - lo) for (lo, hi) in ranges]
        )

        for i in range(0, nBands):
            data[fn][i,:] = (data[fn][i,:] - data[fn][i,:].mean()) / data[fn][i,:].std()

# Data is no a filename (measurement region) -> samples mapping
# Samples are fourier-transform-amplitude frequency-band sums for (193) epochs
# 193 columns (epochs)
# 5 rows (bands)

# Input to k-Means is epochs * measurements x nBands (193 * 27 x 5) matrix
allSamples = []
for k in sorted(data.keys()):
    for v in data[k].transpose():
        allSamples.append(v)
kMeansInput = np.array(allSamples)

model = KMeans(n_clusters = nClusters).fit(np.array(
    sample([row for row in kMeansInput], floor(fitFrac * kMeansInput.shape[0]))
))

predictions = model.predict(kMeansInput).reshape((len(data.keys()), nEpochs))
np.savetxt("misc/kmeansExtendedSpectrum.csv", predictions, delimiter=",")



# Let's do some grouping as well. AKA taking several nBands-dim vectors and concatenating them

groupSize = 5
# Groups per measurement
gpm = floor(nEpochs / groupSize)

groupData = {}
for k in data.keys():
    # Each row a "new" sample
    grps = np.zeros((gpm, groupSize * nBands))
    for i in range(0, gpm):
        grps[i] = data[k][:, i * groupSize : (i + 1) * groupSize].reshape((groupSize * nBands))
    groupData[k] = grps

allGroupSamples = []
for k in sorted(groupData.keys()):
    for v in groupData[k]:
        allGroupSamples.append(v)
groupKMeansInput = np.array(allGroupSamples)

groupModel = KMeans(n_clusters = nClusters).fit(np.array(
    sample([row for row in groupKMeansInput], floor(fitFrac * groupKMeansInput.shape[0]))
))

groupPredictions = groupModel.predict(groupKMeansInput).reshape((len(groupData.keys()), gpm))
np.savetxt("misc/kmeansExtendedSpectrumGrouping.csv", groupPredictions, delimiter=",")