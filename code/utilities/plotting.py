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
            raster[i,j] += samples[
                (samples[ : , 0] >= minx1 + i * x1incr) & \
                (samples[ : , 0] < minx1 + (i + 1) * x1incr) & \
                (samples[ : , 1] >= minx2 + j * x2incr) & \
                (samples[ : , 1] < minx2 + (j + 1) * x2incr)
            ].flatten().shape[0]
            
    return raster