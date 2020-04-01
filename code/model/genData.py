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

#####
#
#   Not really following documentational standards here.
#   This is the failed attempt at generating data with the spectogram of the original data as seed.
#   Failed to interpret the data output, failed to debug the code, moved on to greener pastures
#
#####

def generate_ifft_dataset_type_6():

    epochs = 200
    channels = 25
    n = 2 ** 13
    fs = 16000

    # Jury-rig it
    import matlab_functions as mlf
    import random

    channelData, _ = mlf.getMatlabValues("../_data/matlabData/NPR-075.b11.mat")
    seed_values = np.array([
        # All to same size
        values[0 : 190 * n] for key, values in channelData.items() if "str_lfp" in key or "gp_lfp" in key
    # Rows of length n
    ]).reshape((-1, n))

    samples = np.array(random.sample(list(seed_values), 1000))
    spectogram = np.fft.fft(samples).sum(axis = 0) / 1000.
    
    spectMatrx = np.tile(spectogram, channels).reshape((channels, spectogram.shape[0]))
    epoch = lambda: np.fft.ifft(
        (spectMatrx * 
        np.exp(1j * np.random.uniform(-np.pi, np.pi, size = (channels, n)))) * 
        np.random.uniform(0.5, 1.5, size = (channels, 1)))

    out = np.concatenate([epoch() for _ in np.arange(0, epochs)], axis = 1)
    out = np.array([(row - row.mean()) / row.std() for row in out])
    
    return out, "\n"

#####
#
#   Spectogram-model based data
#   Uses the spectogram 1/f^alpha
#   To this adds A * pdf(f) for some pdf ~ N(mu, sigma) and A
#   Here mu is in the beta-range and sigma is somewhat small
#
#   A is chosen per-epoch with high variance (so all the epochs have similar A)
#   A is then increased or decreased with much lower variance within the epoch (small per-channel variations)
#
#####

def generate_ifft_dataset_type_7():

    epochs = 200
    channels = 25
    n = 2 ** 13
    fs = 16000

    pdf = lambda m, s, x: np.exp(-(x - m) ** 2 / (2 * s ** 2)) / np.sqrt(2 * np.pi * s ** 2)
    spectogram = lambda amp: amp * pdf(21, 5, np.arange(1, n + 1)) + np.arange(1, n+1) ** -1.5

    #####
    #
    #   Try this for context:
    #   plt.plot(np.arange(1, 51), spectogram( 0)[0:50])
    #   plt.plot(np.arange(1, 51), spectogram( 1)[0:50])
    #   plt.plot(np.arange(1, 51), spectogram(-1)[0:50])
    #   plt.show()
    #
    #####

    spectMatrix = lambda amp: np.array([
        spectogram(amp + np.random.uniform(-0.1, 0.1)) for _ in np.arange(0, channels)
    ])
    epoch = lambda amp: np.fft.ifft(
        (spectMatrix(amp) * 
        np.exp(1j * np.random.uniform(-np.pi, np.pi, size = (channels, n)))))

    out = np.concatenate([epoch(np.random.uniform(2, 4)) for _ in np.arange(0, epochs)], axis = 1)
    return out, ""

#####
#
#   As type 7, but more gaussians, more stochasticity
#   I don't remember entirely and I am tired at the time of writing
#
#####

def type_8(
    epochs = 200,
    channels = 25,
    n = 2 ** 13,
    alpha = 1.5,
):
    rang = np.arange(1, 1 + n)
    pfd_normal = lambda mu, sigma, x: np.exp( - (x - mu) ** 2 / ( 2 * sigma ** 2) ) \
        / np.sqrt( 2 * np.pi * sigma ** 2)
    r_u = lambda x, y: np.random.uniform(x, y)

    spectogram = lambda lo, mid, hi: rang ** -alpha \
        + lo * pfd_normal(16.5, 3, rang) \
        + mid* pfd_normal(21,   3, rang) \
        + hi * pfd_normal(25.5, 3, rang)

    spectEpoch = lambda lo, mid, hi, loInter, midInter, hiInter:\
        np.array([ spectogram (
            lo + r_u(-loInter, loInter),
            mid+ r_u(-midInter, midInter),
            hi + r_u(-hiInter, hiInter)
        ) for _ in np.arange(0, channels)])

    epoch = lambda lo, mid, hi, loInter, midInter, hiInter:\
        np.fft.ifft(
            spectEpoch(lo, mid, hi, loInter, midInter, hiInter) * 
            np.exp(1j * np.random.uniform(-np.pi, np.pi, size = (channels, n)))
        )
    
    out = np.concatenate([
        epoch(
            r_u(0.8, 2.1), r_u(0.9, 2.2), r_u(0.8, 2.2), 0.1, 0.1, 0.1
        ) for _ in np.arange(0, epochs)
    ], axis = 1)
    
    return out, ""
