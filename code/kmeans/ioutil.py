from os import mkdir
import matplotlib.pyplot as plt
from centers_dmatrix import weightedDistanceMatrix
import numpy as np

# Saves a 2D numpy array as a color-coded graph
def saveMatrix(matrix, outFile, colorbar = True, label = None):
    plt.imshow(matrix)
    if colorbar:
        plt.colorbar()
    if label != None:
        plt.xlabel(label)
    plt.savefig(outFile)
    plt.clf()

# Saves a 2D numpy array as a .csv file
def saveCsv(matrix, outFile):
    np.savetxt(outFile, matrix, delimiter = ",")

# Ensures the existence of a directory at a given path and returns that path.
# Example usage:
# myOutDirectory = touchDir("makeSureThisExists/")
def touchDir(dirPath):
    try:
        mkdir(dirPath)
    except:
        pass
    return dirPath

# Saves a whole lot of data for the output of a single run with our model.
def saveCustomKexModelOutput(
    parentDir,          # Parent directory to save the output to. Ex waveData/
    childDir,           # Child directory to save the output to. Ex run1/
                        # Would generate directory tree:
                        #   waveData/
                        #       run1/
                        #           predictions/
                        #           centers/
                        #           weighted_distance_matrix/
    predictions,        # The model predictions.
    model,              # The model.
    trainingData,       # The training data.
    epochGroupSize,     # The epoch group size.
    docString           # A string from which a README specifying 
                        #   the parameters for the run is generated.
):
    out = touchDir(parentDir)
    out = touchDir(out + childDir)

    with open(out + "README.txt", "w") as readme:
        readme.write(docString)

    mkpng = lambda epGrSz: "groupSize" + str(epGrSz) + ".png"
    mkcsv = lambda epGrSz: "groupSize" + str(epGrSz) + ".csv"

    # Output predictions
    predictionsOut = touchDir(out + "predictions/")
    saveMatrix(predictions, predictionsOut + mkpng(epochGroupSize), 
        colorbar = False, 
        label = "Class colors are not indicative of class similarity, n classes = " + str(len(np.unique(predictions)))
    )
    predictionsCsv = touchDir(predictionsOut + "csv/")
    saveCsv(predictions, predictionsCsv + mkcsv(epochGroupSize))

    # Output cluster center
    centers = model.cluster_centers_
    centersOut = touchDir(out + "centers/")
    saveMatrix(centers, centersOut + mkpng(epochGroupSize),
        label = "Each row represents a cluster center vector"
    )
    centersCsv = touchDir(centersOut + "csv/")
    saveCsv(centers, centersCsv + mkcsv(epochGroupSize))

    # Output weighted distance matrix
    weightedDistMatr = weightedDistanceMatrix(centers, trainingData, predictions)
    distMatrOut = touchDir(out + "weighted_distance_matrix/")
    saveMatrix(weightedDistMatr, distMatrOut + mkpng(epochGroupSize),
        label = "[i, j] = distance cluster centers i, j / mean intra-cluster distance i"
    )
    distMatrCsv = touchDir(distMatrOut + "csv/")
    saveCsv(weightedDistMatr, distMatrCsv + mkcsv(epochGroupSize))