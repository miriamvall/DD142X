# Dependencies/packages
from sklearn.cluster import KMeans
from random import sample
from math import floor, pi
import numpy as np
import matplotlib.pyplot as plt

# Custom code
import genData
from matlab_functions import getMatlabValues
from fourier import fftEpochsSpecFreq_matrix_and_list
from normalize import normalizedRows
from grouping import groupColumns
from corr import corrcoef
from ioutil import saveCsv, saveMatrix, touchDir, saveCustomKexModelOutput

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
#   Training (which is also testing) data for said model
#   Predictions on training data for said model and training data
#   A parameter documentation string
# 
#   Optional:
#       Dropout         (Remove random-size subset of testing data from training)
#       Normalization   (Currently to 0-mean, 1-std)
#       Epoch grouping  (Several epoch are considered one single sample)
#                       (Heuristic is reduced loss of "over time" information)
#                       (May result in some samples being lost, see grouping.py documentation)
#
def makePredictions(
    allValues,                  # 2D numpy array of values, rows are channels
    loFreq, hiFreq,             # Frequencies to extract
                                #   ** NOT PRECISE DUE TO ROUNDING ERRORS **
                                #   Use fourier.py -> fftEpochSpecFreq to test output frequencies
    corr,                       # True if we want to obtain the correlation coefficients matrix
    epochWidth = 2 ** 13,       # Width of epochs
    FS = 16000,                 # Sampling frequency
    nClusters = 8,              # Amount of clusters for k-Means

    # Additional parameters

    dropout = 0,                # Ratio (0 to 1) of data to exclude during training
    normalize = False,          # If normalization is wanted, set to True
                                #   Affects model training and trainingData output
    epochGroupsSize = 1,        # If set to n > 1, groups epochs into larger vectors
                                # For example, two frequency samples at times t0 and t1 are concatenated
):

    ##      Below: fftValues will be
    #   list of (                   (( List of 2D numpy arrays ))
    #       f1  [e1 e2 e3 ...]      (( Rows are values per-frequency, columns are values per-epoch ))
    #       f2  [e1 e2 e3 ...]      
    #       f3  ...
    #       ...
    #       fn  [e1 e2 e3 ...]
    #   )                           (( The 2D numpy arrays are per-channel ))
    fftValues = [fftEpochsSpecFreq_matrix_and_list(
        row, epochWidth, loFreq, hiFreq, FS
    # [0] is data, [1] is a list of frequencies (useful for debugging)
    )[0] for row in allValues]

    nChannels = len(fftValues)
    # fftValues[0] is "any" 2D numpy matrix
    # fftValues[0].shape[1] is the amount of columns (epochs)
    nEpochs = fftValues[0].shape[1]
    if epochGroupsSize > 1:
        nEpochs = floor(nEpochs / epochGroupsSize)

    if corr:
        corrMatrix = corrcoef(fftValues)
    else:
        corrMatrix = None

    trainingData = []
    for freqXepoch in fftValues:
        # If epoch-grouping:
        if epochGroupsSize > 1:
            # See grouping.py :: test() function for rundown
            for sample in groupColumns(freqXepoch, epochGroupsSize):
                trainingData.append(sample)
        else:
            epochXfreq = freqXepoch.transpose()
            for epoch in epochXfreq:
                trainingData.append(epoch)
    trainingData = np.array(trainingData)

    if normalize:
        trainingData = normalizedRows(trainingData)
    
    model = fitted_kmeans(trainingData, nClusters, dropout)
    predictions = model.predict(trainingData).reshape((nChannels, nEpochs))

    docString = \
        "### MODEL ###" + \
        "\nFREQUENCIES " + str(loFreq) + ":" + str(hiFreq) + \
        "\nEPOCH WIDTH " + str(epochWidth) + \
        "\nSAMPLING FREQUENCY " + str(FS) + \
        "\nCLUSTERS " + str(nClusters) + \
        "\nDROPOUT " + str(dropout) + \
        "\nNORMALIZATION " + str(normalize) + "\n"

    return model, trainingData, predictions, docString, corrMatrix

def runTest():
    values, docStringValues = genData.type_8()
    corr = True
    for epGrSz in [1, 2, 3, 4, 5]:
            model, trainingData, predictions, docStringModel, corrMatrix = \
            makePredictions(
                values,
                12, 32,
                corr,
                epochWidth=2**13,
                dropout = 0.5,
                normalize=False,
                epochGroupsSize=epGrSz,
            )

            corr = False

            saveCustomKexModelOutput(
                "model_output/type_8/",
                "",
                predictions,
                model,
                trainingData,
                epGrSz,
                docStringValues + docStringModel,
                corrMatrix
            )

runTest()