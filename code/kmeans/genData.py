import numpy as np
import matplotlib.pyplot as plt
from math import pi

# Samples a distribution N(meanV, varM)
# Specifically, meanV = [mean, mean, mean, ...] of dimension dims
# varM = var * I where I is the dims * dims identity matrix
### Currently unnecessarily complicated for its' output
### Can be easily modified to accomodate covariances
def makeMeanAndCovar(mean, var, dims = 4):
    meanOut = np.zeros((dims)) + mean
    covarOut = np.eye((dims)) * var
    return np.random.multivariate_normal(meanOut, covarOut)

# Generates time increments for T = length, dt = 1/fs
# Outputs a 2D numpy array with "dims" equal rows of such time increments
def makeTimes(length, fs, dims = 4):
    return np.tile(np.linspace(0, length, fs * length), (dims, 1))

# Sine over an array (any dimension) with a specified frequency and phase
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

# Generates sine-wave based data for testing use
# Also returns a documentation string for README
# TYPE 1 "Identically Distributed Waves"
#   For all frequencies, over all channels, the amplitudes are identically distributed
#   Each amplitude is ~ N(mean, var)
#   See below for per-parameter documentation
# NOTES
#   Higher variance should result in less synchronization
#   This might hold less true: 
#       For large ranges of frequencies
#       If the data is normalized
def generateWaveDataset_type_1(
    mean = 0,       # Mean for each amplitude
    var = 1,        # Variance for each amplitude
    channels = 25,  # The amount of channels to generate data for
    freqLo = 1,     # Lowest frequency to generate data for (inclusive)
    freqHi = 51,    # Highest frequency to generate data for (exclusive)
    length = 100,   # Length of time in seconds for which to simulate data
    fs = 16000      # Sampling frequency in Hz
):
    values = np.zeros((channels, length * fs))
    for freq in range(freqLo, freqHi):
        values += genWaves(mean, var, channels, freq, length, fs)
    
    docString = \
        "### DATA TYPE 1 Identically Distributed Waves ###" + \
        "\nMEAN " + str(mean) + \
        "\nVARIANCE " + str(var) + \
        "\nCHANNELS " + str(channels) + \
        "\nLOW FREQUENCY " + str(freqLo) + \
        "\nHIGH FREQUENCY " + str(freqHi) + \
        "\nLENGTH (SECONDS) " + str(length) + \
        "\nSAMPLING FREQUENCY " + str(fs) + "\n"

    return values, docString

# Generates one large multivariate gaussian
# Also returns a documentation string for README
# TYPE 2 "Multivariate Gaussian"
#   A 2D numpy array of size channels * (length * fs)
#   Each input value is sampled from N(mean, var)
# NOTES
#   This should show NO NOTICEABLE synchronization
def generateDataset_type_2(
    mean = 0,
    var = 1,
    channels = 25,
    length = 100,
    fs = 16000
):
    docString = \
        "### DATA TYPE 2 Multivatiate Gaussian ###" + \
        "\nMEAN " + str(mean) + \
        "\nVARIANCE " + str(var) + \
        "\nCHANNELS " + str(channels) + \
        "\nLENGTH (SECONDS) " + str(length) + \
        "\nSAMPLING FREQUENCY " + str(fs) + "\n"
    return np.random.normal(mean, var, (channels, length * fs)), docString

# Generates sine-wave based data for testing use
# Also returns a documentation string for README
# TYPE 3 "Identically Distributed Waves of Two Classes"
#   Similar to Type 1
#   There are two classes of waves; those in the beta-range [12Hz - 30Hz] and others
#   Amplitude mean and variance is specified separately for these
# NOTES
#   Allows beta-frequencies to be increased or decreased independently of non-beta frequencies
#   Synchronization should be easily produced, and lowered for increasing beta-variance
#   Certain configurations might increase beta-synchronization
#   For example; should non-beta mean greatly exceed beta mean and normalization be applied in the model,
#       then variance of beta-frequencies should be expected to reduce drastically
def generateWaveDataset_type_3(
    mean = 0,       # Mean for each amplitude (non-beta)
    var = 1,        # Variance for each amplitude (non-beta)
    meanBeta = 0,   # Mean for each amplitude (beta)
    varBeta = 1,    # Variance for each amplitude (beta)
    channels = 25,  # The amount of channels to generate data for
    freqLo = 1,     # Lowest frequency to generate data for (inclusive)
    freqHi = 51,    # Highest frequency to generate data for (exclusive)
    length = 100,   # Length of time in seconds for which to simulate data
    fs = 16000      # Sampling frequency in Hz
):
    values = np.zeros((channels, length * fs))
    nonBeta = [x for x in range(freqLo, freqHi) if x < 12 or x > 30]
    beta    = [x for x in range(freqLo, freqHi) if x not in nonBeta]
    for freq in nonBeta:
        values += genWaves(mean, var, channels, freq, length, fs)
    for freq in beta:
        values += genWaves(meanBeta, varBeta, channels, freq, length, fs)

    docString = \
        "### DATA TYPE 3 Identically Distributed Waves of Two Classes ###" + \
        "\nMEAN " + str(mean) + \
        "\nVARIANCE " + str(var) + \
        "\nMEAN BETA " + str(meanBeta) + \
        "\nVARIANCE BETA " + str(varBeta) + \
        "\nCHANNELS " + str(channels) + \
        "\nLOW FREQUENCY " + str(freqLo) + \
        "\nHIGH FREQUENCY " + str(freqHi) + \
        "\nLENGTH (SECONDS) " + str(length) + \
        "\nSAMPLING FREQUENCY " + str(fs) + "\n"

    return values, docString

# Generates sine-wave based data for testing use
# Also returns a documentation string for README
# TYPE 4 "Waves Distributed by Linearly Decreasing Mean over Frequency"
#   Over all channels, the amplitudes are identically distributed
#   Each amplitude is ~ N(maximum - f * slope, var) where f is the frequency for the wave
# NOTES
#   Should more closely resemble the real situation of local field potentials,
#       if I recall correctly
def generateWaveDataset_type_4(
    maximum = 25,   # Maximum mean value
    slope = 0.5,      # Mean will decrease by this amount as f increases
                    #   Could technically be < 0 for linearly increasing data
    var = 1,        # Variance for each amplitude
    channels = 25,  # The amount of channels to generate data for
    freqLo = 1,     # Lowest frequency to generate data for (inclusive)
    freqHi = 51,    # Highest frequency to generate data for (exclusive)
    length = 100,   # Length of time in seconds for which to simulate data
    fs = 16000      # Sampling frequency in Hz
):
    values = np.zeros((channels, length * fs))
    for freq in range(freqLo, freqHi):
        values += genWaves(maximum - freq * slope, var, channels, freq, length, fs)
    
    docString = \
        "### DATA TYPE 4 Waves Distributed by Linearly Decreasing Mean over Frequency ###" + \
        "\nMAX " + str(maximum) + \
        "\nSLOPE " + str(slope) + \
        "\nVARIANCE " + str(var) + \
        "\nCHANNELS " + str(channels) + \
        "\nLOW FREQUENCY " + str(freqLo) + \
        "\nHIGH FREQUENCY " + str(freqHi) + \
        "\nLENGTH (SECONDS) " + str(length) + \
        "\nSAMPLING FREQUENCY " + str(fs) + "\n"

    return values, docString

# Generates sine-wave based data for testing use
# Also returns a documentation string for README
# TYPE 5 "Waves of Two Classes Distributed by Linearly Decreasing Mean over Frequency"
#   Over all channels, the amplitudes are identically distributed
#   Each amplitude is ~ N(maximum - f * slope, X) where f is the frequency for the wave
#   For frequencies [12Hz - 30Hz] X = varBeta; for others X = var
def generateWaveDataset_type_5(
    maximum = 25,   # Maximum mean value
    slope = 0.5,      # Mean will decrease by this amount as f increases
                    #   Could technically be < 0 for linearly increasing data
    var = 1,        # Variance for each amplitude (non-beta)
    varBeta = 1,    # Variance for each amplitude (beta)
    channels = 25,  # The amount of channels to generate data for
    freqLo = 1,     # Lowest frequency to generate data for (inclusive)
    freqHi = 51,    # Highest frequency to generate data for (exclusive)
    length = 100,   # Length of time in seconds for which to simulate data
    fs = 16000      # Sampling frequency in Hz
):
    values = np.zeros((channels, length * fs))
    for freq in range(freqLo, freqHi):
        values += genWaves(maximum - freq * slope, var, channels, freq, length, fs)
    
    docString = \
        "### DATA TYPE 5 Waves of Two Classes Distributed by Linearly Decreasing Mean over Frequency ###" + \
        "\nMAX " + str(maximum) + \
        "\nSLOPE " + str(slope) + \
        "\nVARIANCE " + str(var) + \
        "\nVARIANCE BETA " + str(varBeta) + \
        "\nCHANNELS " + str(channels) + \
        "\nLOW FREQUENCY " + str(freqLo) + \
        "\nHIGH FREQUENCY " + str(freqHi) + \
        "\nLENGTH (SECONDS) " + str(length) + \
        "\nSAMPLING FREQUENCY " + str(fs) + "\n"

    return values, docString