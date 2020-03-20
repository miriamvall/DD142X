import numpy as np
import matplotlib.pyplot as plt
from math import pi

# Samples a distribution N(meanV, varM)
# Specifically, meanV = [mean, mean, mean, ...] of dimension dims
# varM = var * I where I is the dims * dims identity matrix
def makeMeanAndCovar(mean, var, dims = 4):
    meanOut = np.zeros((dims)) + mean
    covarOut = np.eye((dims)) * var
    return np.random.multivariate_normal(meanOut, covarOut)

# Generates time increments for T = length, dt = 1/fs
# dims equal 1D numpy arrays of this form make up a 2D numpy array with dims rows
def makeTimes(length, fs, dims = 4):
    return np.tile(np.linspace(0, length, fs * length), (dims, 1))

# Sine over an array with a specified frequency
def sines(values, freq = 1., phase = 0.):
    return np.sin(values * 2. * pi * freq + phase)

# Generates wave data
# The amplitudes of the waves are sampled from a multivariate gaussian
# There are dims independent waves
# The waves have frequency freq
# The wave lasts for length time with fs sampling frequency
# If an outString is given, the waves are saved as a figure
def genWaves(mean, var, dims, freq, length, fs, outString = None, phase = 0.):
    amps = makeMeanAndCovar(mean, var, dims)
    times = makeTimes(length, fs, dims)
    wavesUnscaled = sines(times, freq, phase)
    waves = (wavesUnscaled.transpose() * amps).transpose()
    if outString != None:
        for wave in waves:
            plt.plot(wave, 'r', alpha=0.1)
        plt.savefig(outString + ".png")
        plt.clf()
    return waves