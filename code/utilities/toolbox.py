import numpy as np
import matplotlib.pyplot as plt

def aprx_pdf_1d(values, n_buckets = 1000, log2 = True):
    lo = np.min(values)
    hi = np.max(values)
    incr = (hi - lo) / n_buckets
    ys = np.array([
        ((values >= lo + idx * incr) & (values <= lo + (idx + 1) * incr)).sum()
        for idx in np.arange(0, n_buckets)
    ])
    xs = np.array([
        lo + idx * incr for idx in np.arange(0, n_buckets)
    ])
    if log2:
        yslog2 = np.log2(ys + 1)
        return xs, ys / ys.sum(), yslog2 / yslog2.sum()
    return xs, ys

def raster_imshow(values):
    minx1 = np.min(values[ : , 0])
    minx2 = np.min(values[ : , 1])
    maxx1 = np.max(values[ : , 0])
    maxx2 = np.max(values[ : , 1])

    extent = [minx2, maxx2, maxx1, minx1]

    raster = np.zeros((
        int(round(maxx1 - minx1, 0) + 1), 
        int(round(maxx2 - minx2, 0) + 1)
    ))

    for x1, x2 in values.copy() - np.array((minx1, minx2)):
        raster[
            int(round(x1, 0)), 
            int(round(x2, 0))
        ] += 1

    plt.imshow(raster, extent = extent, aspect = 'auto', cmap = 'gist_yarg')
    plt.colorbar()