import numpy as np
import math

# Groups columns of a matrix into larger arrays.
# len(values) % groupSize values are ignored on the tail(s) of values, as they don't make up a whole group.
# values should be a 2D-array with values on rows and epochs on columns.
# groupSize should be an integer.
# The new outputs are rows in the output matrix
def groupColumns(values, groupSize):
    transpose = values.transpose()
    nGroups = math.floor(transpose.shape[0] / groupSize)
    newSize = transpose.shape[1] * groupSize
    out = np.zeros((nGroups, newSize))
    for i in range(0, nGroups):
        # Selects rows
        # Ex groupSize = 2
        #   Selects 0:2, 2:4, 4:6...
        out[i,:] = transpose[i * groupSize : (i + 1) * groupSize, :].reshape((newSize))
    return out

def test():
    epochs = np.array((
        1, 2, 3, 4, 5,
        2, 3, 4, 5, 6,
        3, 4, 5, 6, 7,
        4, 5, 6, 7, 8
    )).reshape((4, 5))
    print(groupColumns(epochs, 2))
    #   Output:
    #   [[1, 2, 3, 4, 2, 3, 4, 5]
    #    [3, 4, 5, 6, 4, 5, 6, 7]]
