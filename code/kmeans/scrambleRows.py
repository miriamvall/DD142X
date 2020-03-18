import numpy as np

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
        shift = (shift + intensity) % rows
    return out

def test():
    b = np.array((
        10, 11, 12, 13, 14, 15, 16, 17,
        20, 21, 22, 23, 24, 25, 26, 27,
        30, 31, 32, 33, 34, 35, 36, 37,
        40, 41, 42, 43, 44, 45, 46, 47
    )).reshape((4, 8))
    print(b)
    print(scrambleRows(b, intensity = 1))
#  [[10. 21. 32. 43. 14. 25. 36. 47.]
#   [20. 31. 42. 13. 24. 35. 46. 17.]
#   [30. 41. 12. 23. 34. 45. 16. 27.]
#   [40. 11. 22. 33. 44. 15. 26. 37.]]
