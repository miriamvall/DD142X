import numpy as np
from random import randint

# Shifts the data in values.
# values should be a 2D numpy array.
# The output contains all the same data as values, but reordered.
# Each value in values is moved "up" in the matrix by an amount equal to the index of its column.
# A value that goes "above" the matrix loops around to the "bottom".
# See the example in "test".
#
#   This should ruin the epoch-results of our k-Means algorithm.
#   If this doesn't ruin any synchronization, we have a problem.
#
def scrambleRows(values, intensity = 1):
    out = np.zeros(values.shape)
    cols = values.shape[1]
    rows = values.shape[0]
    shift = 0
    for col in range(0, cols):
        part1 = values[0:shift      : , col]
        part2 = values[shift:rows   : , col]
        out[:,col] = np.concatenate((part2, part1))
        if intensity == "random":
            shift = randint(0, rows - 1)
        else:
            shift = (shift + intensity) % rows
    return out

# Generates trash data. Woop.
def trashData(values):
    out = np.zeros(values.shape)
    for i in range(0, values.shape[0]):
        for j in range(0, values.shape[1]):
            out[i, j] = randint(0, 100)
    return out

def pickAndMix(values):
    cols = values.shape[1]
    rows = values.shape[0]
    switches = rows * 5
    out = values.copy()
    for col in range(0, cols):
        for iter in range(0, switches):
            idx1 = randint(0, rows - 1)
            idx2 = randint(0, rows - 1)
            tmp = out[idx1, col]
            out[idx1, col] = out[idx2, col]
            out[idx2, col] = tmp
    return out


def test():
    b = np.array((
        10, 11, 12, 13, 14, 15, 16, 17,
        20, 21, 22, 23, 24, 25, 26, 27,
        30, 31, 32, 33, 34, 35, 36, 37,
        40, 41, 42, 43, 44, 45, 46, 47,
        50, 51, 52, 53, 54, 55, 56, 57
    )).reshape((5, 8))
    print(b)
    print(scrambleRows(b, 1))
    print(scrambleRows(b, 3))
    print(pickAndMix(b))

