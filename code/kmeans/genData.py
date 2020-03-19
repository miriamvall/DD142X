import numpy as np
import matplotlib.pyplot as plt
from math import pi

def makeMeanAndCovar(mean, var, dims = 4):
    meanOut = np.zeros((dims)) + mean
    covarOut = np.eye((dims)) * var
    return np.random.multivariate_normal(meanOut, covarOut)

def makeTimes(length, fs, dims = 4):
    return np.tile(np.linspace(0, length, fs * length), (dims, 1))

def sines(values, freq = 1.):
    return np.sin(values * 2. * pi * freq)

def genWaves(mean, var, dims, freq, length, fs, outString = None):
    amps = makeMeanAndCovar(mean, var, dims)
    times = makeTimes(length, fs, dims)
    wavesUnscaled = sines(times, freq)
    waves = (wavesUnscaled.transpose() * amps).transpose()
    if outString != None:
        for wave in waves:
            plt.plot(wave, 'r', alpha=0.1)
        plt.savefig(outString + ".png")
        plt.clf()
    return waves