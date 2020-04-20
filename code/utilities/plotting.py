import numpy as np

# Hard to understand but efficient function.
# Written in a "pythonic" (simpler) rather than "numpythonic" way, takes many many MANY times longer.
def rasterize(samples, side = 40):
    # Find range of values; input is N x 2 numpy array
    maxx1 = np.max(samples[ : , 0])
    minx1 = np.min(samples[ : , 0])
    maxx2 = np.max(samples[ : , 1])
    minx2 = np.min(samples[ : , 1])
    
    # Decides the "pixels"
    x1incr = (maxx1 - minx1) / side
    x2incr = (maxx2 - minx2) / side
    raster = np.zeros((side, side))
    
    # Count amount of samples belonging to each pixel
    for i in range(0, side):
        for j in range(0, side):
            # 
            #    Uses numpy boolean indexing
            #
            count = samples[
                (samples[ : , 0] >= minx1 + i * x1incr) & \
                (samples[ : , 0] < minx1 + (i + 1) * x1incr) & \
                (samples[ : , 1] >= minx2 + j * x2incr) & \
                (samples[ : , 1] < minx2 + (j + 1) * x2incr)
            ]
            raster[i,j] += count.shape[0]
            
    return raster

# Similar to rasterize, but attempt to respect range of data.
# Note that all data is shifted so that min is at index 0 for both dimensions.
def raster_withbounds(inputs):
    points = inputs[:, 0:2].copy().astype(int)
    raster = np.zeros((
        np.max(points[:, 0]) - np.min(points[:, 0]) + 1,
        np.max(points[:, 1]) - np.min(points[:, 1]) + 1
    ))
    
    points[:, 0] -= np.min(points[:, 0])
    points[:, 1] -= np.min(points[:, 1])
    
    for p in points:
        raster[p[0], p[1]] += 1
    return raster

def simultaneous_raster_withbounds(inputs1, inputs2, log2 = False):
    points1 = inputs1[:, 0:2].copy().astype(int)
    points2 = inputs2[:, 0:2].copy().astype(int)

    maxx1 = np.max((
        np.max(points1[:, 0]),
        np.max(points2[:, 0])
    ))
    minx1 = np.max((
        np.min(points1[:, 0]),
        np.min(points2[:, 0])
    ))
    maxx2 = np.max((
        np.max(points1[:, 1]),
        np.max(points2[:, 1])
    ))
    minx2 = np.max((
        np.min(points1[:, 1]),
        np.min(points2[:, 1])
    ))

    points1[ : , 0] -= minx1
    points1[ : , 1] -= minx2
    points2[ : , 0] -= minx1
    points2[ : , 1] -= minx2

    raster = np.zeros((
        maxx1 - minx1 + 1,
        maxx2 - minx2 + 1,
        3
    ))

    for x1, x2 in points1:
        raster[x1, x2, 0] += 1
    for x1, x2 in points2:
        raster[x1, x2, 2] += 1

    if log2:
        rasterlog2 = raster.copy()
        rasterlog2[ : , : , 0] = np.log2(rasterlog2[ : , : , 0] + 1)
        rasterlog2[ : , : , 2] = np.log2(rasterlog2[ : , : , 2] + 1)
        return raster, rasterlog2

    return raster