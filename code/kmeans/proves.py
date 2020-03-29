from sklearn.cluster import KMeans
import numpy as np
from random import sample
from math import floor
import matplotlib.pyplot as plt

from matlab_functions import getMatlabValues
from fourier import fftEpochsSpecFreq
from normalize import normalizedRows
from grouping import groupColumns
from scrambling import scrambleRows, trashData, pickAndMix

# Returns a k-Means model fitted to the input values.
# The values should be a 2D numpy array, where the rows are considered inputs to k-Means.
# Set dropout to a number between 0 and 1 to exclude that portion of the data from training.
def fitted_kmeans(values, Nclusters, dropout = 0):
    return KMeans(n_clusters = Nclusters).fit(
        np.array(sample(list(values), floor((1 - dropout) * values.shape[0])))
    )


# Takes lot of input parameters. See documentation.
#
# Produces:
#   A k-Means model
#   testing data for said model
#   predictions on training data for said model and training data
# 
# Currently exclusively adapted to our specific method of:
#   .mat data ->
#   channel data ->
#   epochs (of raw channels data) ->
#   epochs (fft of previous)
# 
#   Optional:
#       Dropout
#       Normalization
#       Epoch grouping 
#
def makePredictions(
    allValues,                      # Dictionary channel : values
                                    # Ex:   "str_lfp1" : np.array((0.1, 0.2, ....))
    channelPatterns,                # Example: [str_lfp, gp_lfp]; only channels that match a pattern
    loFreq, hiFreq,                 # Frequencies to extract
    epochWidth = 2 ** 13,           # Width of epochs
                                    #   ** NOT PRECISE DUE TO ROUNDING ERRORS **
                                    #   Use fourier.py -> fftEpochSpecFreq to test output frequencies
    FS = 16000,                     # Sampling frequency
    nClusters = 8,                  # Amount of clusters for k-Means

    # Additional parameters

    dropout = 0,                # Ratio (0 to 1) of data to exclude during training
    normalize = False,          # If normalization is wanted, set to True
                                #   Affects model training and trainingData output
    epochGroupsSize = 1,        # If set to n > 1, groups epochs into larger vectors
                                # For example, two frequency samples at times t0 and t1 are concatenated
    scramble = (False, 0)       # One method of "destroying" the input data. Should kill any and all results.
                                # Boolean for yes/no, integer for intensity.
):

    values = {}
    # Clean out non- str_lfp, gp_lfp
    for pattern in channelPatterns:
        for key in allValues:
            if pattern in key:
                 values[key] = allValues[key]
    # As list of tuples (k, v) instead of dict k : v
    values = [(key, value) for key, value in values.items()]

    # Output format:
    # list(                     [
    #   tuple(                      (
    #       channel                     "str_lfp1",
    #       list(                       [
    #           tuple(                      (
    #               frequency                   11.71, # Hz
    #               values                      np.array(0.1, 0.2, 0.3...)
    #           )                           )
    #       )                           ]
    #    )                          )
    # )                         ]
    fftValues = [
        (key, fftEpochsSpecFreq(value, epochWidth, loFreq, hiFreq, FS)) 
        for (key, value) in values
    ]
    # For the nested values, "columns" are epochs
    # "Rows" are values "per frequency"
    # E.g., each "column" is a point of data

    nChannels = len(fftValues)
    nEpochs = len(fftValues[0][1][0][1]) # Use above graph to understand what's going on here
    if epochGroupsSize > 1:
        nEpochs = floor(nEpochs / epochGroupsSize)

    trainingData = []
    for (channel, freqValList) in fftValues:

        freqXepoch = np.array([val for (_, val) in freqValList])

        # If epoch-grouping:
        if epochGroupsSize > 1:
            for sample in groupColumns(freqXepoch, epochGroupsSize):
                trainingData.append(sample)
        # Else:
        else:
            epochXfreq = freqXepoch.transpose()
            for epoch in epochXfreq:
                trainingData.append(epoch)

    trainingData = np.array(trainingData)

    # Normalization, if any
    if normalize:
        trainingData = normalizedRows(trainingData)

    testingData = trainingData.copy()

    # Scrambling of data
    if scramble[0] == "trash":
        trainingData = trashData(trainingData)
    elif scramble[0] == "pickAndMix":
        trainingData == pickAndMix(trainingData)
    elif scramble[0]:
        trainingData = scrambleRows(trainingData, scramble[1])
    
    model = fitted_kmeans(trainingData, nClusters, dropout)
    predictions = model.predict(testingData).reshape((nChannels, nEpochs))

    return model, testingData, predictions

# Saves a 2D numpy array as a color-coded graph
def saveMatrix(matrix, outFile):
    plt.imshow(matrix)
    plt.colorbar()
    plt.savefig(outFile)
    plt.clf()

# Calculates the distances between arrays and returns a distance (norm) matrix
# The input should be a 2D numpy array, where rows are arrays for which you want distances
def distanceMatrix(arrays):
    r = arrays.shape[0]
    out = np.zeros((r, r))
    for i in range(0, r):
        for j in range(0, r):
            out[i, j] = np.linalg.norm(arrays[i] - arrays[j])
    return out

# Returns an array of the average distance (norm) from cluster centers for some data.
# The amount of rows in data (samples) should be equal to the amount of predictions.
# Centers should be a 2D numpy array where each row is a cluster center.
# Data should be a 2D numpy array where each row is a sample.
# Predictions should be a 1D numpy array assigning each sample a class (distinct cluster)
def centerDistances(centers, data, predictions):
    nClasses = centers.shape[0]
    byPrediction = [
        [] for i in range(0, nClasses)
    ]
    # Group by class
    for (row, prediction) in zip(data, predictions):
        byPrediction[prediction].append(row)
    for i in range(0, nClasses):
        # Broadcast subtraction of centers[i] over all class i samples
        if len(byPrediction[i]) > 0:
            byPrediction[i] =  np.array(byPrediction[i]) - centers[i]
        else:
            # If no sample assigned this class, make arbitrarily large
            # Must have at least two elements
            print("No assignment indicator class " + str(i))
            byPrediction[i] = np.array((1e10, 1e10))
        # Map to norms
        byPrediction[i] = np.array([
            np.linalg.norm(row) for row in byPrediction[i]
        # Take mean
        ]).mean()
    return np.array(byPrediction)


def main():

    allValues = getMatlabValues("../_data/matlabData/NPR-075.b11.mat")

    configurations = [
        ("normal", (False, None)),
        ("pickAndMix", ("pickAndMix", None)),
        ("randomScrambled", (True, "random")),
        ("scrambled", (True, 17))
    ]

    # Directory for output of a certain data scrambling configuration
    for (outDir, scrambleConf) in configurations:
        # Several epoch-groupsizes for each configuration
        for i in [1, 2, 3, 4, 5]:
            model, testingData, predictions = makePredictions(
                allValues,
                ["str_lfp", "gp_lfp"],
                12, 32,
                epochWidth=2**13,
                dropout = 0.5,
                normalize=True,
                epochGroupsSize=i,
                scramble=scrambleConf,
                nClusters=4
            )
            out = "k-meansFigures/" + outDir + "/"
            saveMatrix(predictions, out + "groupSize" + str(i) + ".png")

            predictionsAs1D = predictions.reshape((
                predictions.shape[0] *
                predictions.shape[1]
            ))
            centers = model.cluster_centers_
            nC = len(centers)
            
            centDist = centerDistances(centers, testingData, predictionsAs1D)
            distMatr = distanceMatrix(centers)
            weightedDistMatr = distMatr / centDist # Broadcasts

            saveMatrix(weightedDistMatr, out + "centerWeightedGroupSize" + str(i) + ".png")
            saveMatrix(centers, out + "allCentersGroupSize" + str(i) + ".png")

            print("Finished " + outDir + " " + str(i))
            
main()