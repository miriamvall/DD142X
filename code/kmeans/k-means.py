from sklearn.cluster import KMeans
import numpy as np
from random import sample
from math import floor
import matplotlib.pyplot as plt

from matlab_functions import *
from fourier import *
from normalize import *
from grouping import *

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
#   training data for said model
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
    fileName,                       # A .mat file
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
    epochGroupsSize = 1         # If set to n > 1, groups epochs into larger vectors
                                # For example, two frequency samples at times t0 and t1 are concatenated
):
    # Dictionary channel : values
    # Ex:   "str_lfp1" : np.array((0.1, 0.2, ....))
    allValues = getMatlabValues(fileName)
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
    
    model = fitted_kmeans(trainingData, nClusters, dropout)
    predictions = model.predict(trainingData).reshape((nChannels, nEpochs))

    return model, trainingData, predictions

def main():

    model, trainingdata, predictions = makePredictions(
        "../_data/matlabData/NPR-075.b11.mat",
        ["str_lfp", "gp_lfp"],
        12, 32,
        epochWidth=2**13,
        dropout = 0.7,
        normalize=True,
        epochGroupsSize=2
    )

    plt.imshow(predictions)
    plt.colorbar()
    plt.show()

main()