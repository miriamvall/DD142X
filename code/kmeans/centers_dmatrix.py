import numpy as np

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
def centerDistances(centers, data, predictions, verbose = False):
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
            if verbose:
                print("No assignment indicator class " + str(i))
            byPrediction[i] = np.array((1e15, 1e15))
        # Map to norms
        byPrediction[i] = np.array([
            np.linalg.norm(row) for row in byPrediction[i]
        # Take mean
        ]).mean()
    return np.array(byPrediction)

# Cluster center distances weighted by within-cluster mean distance
def weightedDistanceMatrix(centers, testingData, predictions):
    predictionsAs1D = predictions.reshape((
        predictions.shape[0] *
        predictions.shape[1]
    ))
    centDist = centerDistances(centers, testingData, predictionsAs1D)
    distMatr = distanceMatrix(centers)
    return distMatr / centDist