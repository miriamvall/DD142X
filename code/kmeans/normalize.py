import numpy as np

# Returns a normalized array by { values = (values - mean(values)) / stdiv(values) }
# Does not modify the original array.
def normalizedArray(values):
    return (values - np.mean(values)) / np.std(values)

# As normalizedArray, but for each seperate row in a matrix (2D-array)
def normalizedRows(values):
    return np.array([
        (row - np.mean(row)) / np.std(row) for row in values
    ])

# A quick demonstration
def testNormalize():
    arr = np.array((1, 2, 3))
    matrx = np.array((1, 2, 3, 1, 2, 3, 1, 2, 3)).reshape((3, 3))
    print(arr)
    print(matrx)
    print(normalizedArray(arr))
    print(normalizedRows(matrx))