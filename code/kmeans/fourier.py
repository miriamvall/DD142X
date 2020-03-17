import numpy as np
import math

import matlab_functions

# Takes a numpy array "values" and returns fft of epochs (slices) of length "width" of that array
# Output rows are frequencies, columns are epochs
# Only returns amplitudes abs(fft(values, width))
def fftEpochs(values, width):
    nEpochs = math.floor(values.shape[0] / width)
    epochs = np.zeros((width, nEpochs))
    for i in range(0, nEpochs):
        toFft = np.fft.fft(values[i * width : (i + 1) * width])
        toAmp = np.abs(toFft)
        epochs[:,i] = toAmp
    return epochs

# Takes a numpy array "values" and returns fft of epochs (slices) of length "width" of that array
# Output rows are frequencies, columns are epochs
# Only returns amplitudes abs(fft(values, width))
# Returns only the values between (approximately) the specified frequencies, as calculated using FS = sampling frequency
#   When using, adjusting "lo" and "hi" based on output could be wise.
# The values are returned in a list of tuples (f, arr) where f is the frequency and arr are the values.
def fftEpochsSpecFreq(values, width, lo, hi, FS):
    dfft = fftEpochs(values, width)
    loIndex = math.floor(width * lo / FS)
    hiIndex = math.floor(width * hi / FS)
    out = []
    for i in range(loIndex, hiIndex):
        out.append((i * FS / width, dfft[i,:]))
    return out

# As fftEpochsSpecFreq, but returns a 2D-array and a list of frequencies instead
def fftEpochsSpecFreqAlt(values, width, lo, hi, FS):
    dfft = fftEpochs(values, width)
    loIndex = math.floor(width * lo / FS)
    hiIndex = math.floor(width * hi / FS)
    outMatrix = np.array([
        dfft[i,:] for i in range(loIndex, hiIndex)
    ])
    outFreqs = [i * FS / width for i in range (loIndex, hiIndex)]
    return outMatrix, outFreqs

# Quick sanity check
# Amplitudes should peak at f = 10, f = 20 Hz
def test():
    width = 2**13
    lo = 10
    hi = 22
    FS = 16000
    # Example with four epochs
    times = np.array(range(0, width * 4)) / FS
    # Note amplitudes, frequencies
    signal1 = 10 * np.sin(2 * math.pi * 10 * times)
    signal2 = 20 * np.cos(2 * math.pi * 20 * times)
    out = fftEpochsSpecFreq(signal1 + signal2, width, lo, hi, FS)
    for t in out:
        print(t)
    matr, freq = fftEpochsSpecFreqAlt(signal1 + signal2, width, lo, hi, FS)
    print(matr)
    print(freq)