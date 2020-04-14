import numpy as np

# Fourier Feature Vector(s)
# Now auto-resizes: Will not work on incompatible input size
#
#   Returns a DFT-based feature vector for the input
#   1. Resizes the input 2D numpy array to rows of epoch_size
#   2. Finds frequency indices in range [lo, hi)
#   3. Pad each row (epoch) with zeroes until epoch is of length fft_n
#   4. Applies DFT to each row
#   5. Converts output to amplitude spectrum (absolute value)
#   6a. Returns amplitude spectrum for each epoch where frequencies are contained in [lo, hi)
#   6b. Also returns a list that serves as a lookup table for output frequencies
#   Ex: If ffv[1][idx] == f; ffv[k][idx] is amplitude for frequency f for any k
#
def ffv(
    xs,                     # Input values: 2D numpy array
    Fs = 16000.,            # Sampling frequency
    epoch_size = 2 * 11,    # Epoch size
    fft_n = 2 ** 14,        # Increase results in greater dimensionality output
    lo = 12,                # Lowest frequency to include in output (incl.)
    hi = 30                 # Highest frequency to include in output (excl.)
):
    # Reshape to epochs
    fft_in = xs.reshape((-1, epoch_size))
    # Frequencies of FFT
    frqs = np.fft.fftfreq(fft_n, 1./Fs)
    # Find low index, high index
    loidx = np.where(frqs >= lo)[0][0]
    hiidx = np.where(frqs >  hi)[0][0]
    # np.abs -> Amplitudes
    ffts = np.abs(
        # Select only matching frequencies in range
        np.fft.fft(fft_in, fft_n)[ : , loidx : hiidx]
    )

    # Return feature vectors, frequency lookup table
    return ffts, frqs[loidx : hiidx]

# Fourier Power Feature Vector
# Equivalent to ffv( < args > )[0] ** 2, ffv( < args > )[1] in output
def fpfv(
    xs,                     # Input values: 2D numpy array
    Fs = 16000.,            # Sampling frequency
    epoch_size = 2 * 11,    # Epoch size
    fft_n = 2 ** 14,        # Increase results in greater dimensionality output
    lo = 12,                # Lowest frequency to include in output (incl.)
    hi = 30                 # Highest frequency to include in output (excl.)
):
    v, f = ffv(xs, Fs, epoch_size, fft_n, lo, hi)
    return v ** 2, f

# As Probability Density Function
# Takes a 2D numpy array as input
# Each input row x = [x1, x2, x3...]
# has output row y = x / x.sum()
# Convenience function
def asPDF(xs):
    return np.array([
        row / row.sum() for row in xs
    ])

# Returns the normalized Shannon Entropy of each row, as normalized to PDF
# The output entropy samples are normalized to [0, 1]
# Takes an input 2D numpy array
# Returns a 1D numpy array where output[i] is the Shannon Entropy of input[i] / input[i].sum()
def asEntropy(xs):
    pdf = asPDF(xs)
    N = xs.shape[1]
    return - np.array([
        np.array([
            p * np.log2(p) for p in row
        ]).sum() for row in pdf 
    ]) / np.log2(N)

# Spectral Entropy Vector
# Returns a vector of Spectral Entropy* for some input
# * Equivalent to asEntropy ( fpfv ( < args > )[0] )
def sev(
    xs,                     # Input values: 2D numpy array
    Fs = 16000.,            # Sampling frequency
    epoch_size = 2 * 11,    # Epoch size
    fft_n = 2 ** 14,        # Increase results in greater dimensionality output
    lo = 12,                # Lowest frequency to include in output (incl.)
    hi = 30                 # Highest frequency to include in output (excl.)
):
    return asEntropy(
        fpfv(xs, Fs, epoch_size, fft_n, lo, hi)[0]
    )

# Almost equivalent to ffv; equivalent in output
# memory_var limits the amount of rows that can be input to np.fft.fft(<input>, fft_n)
# This can be particularly useful for large inputs that require very large memory allocations
# For other purposes, this functions is seemingly slower
def alt_ffv(
    xs,
    Fs = 16000.,
    epoch_size = 2 ** 11,
    fft_n = 2 ** 14,
    lo_f = 12,
    hi_f = 30,
    memory_var = 2 ** 14
):
    fft_in = xs.reshape((-1, epoch_size))
    frqs = np.fft.fftfreq(fft_n, 1. / Fs)
    lo = np.where(frqs > lo_f)[0][0]
    hi = np.where(frqs > hi_f)[0][0]
    
    # Helper recursive function
    def fftrec(xxs):
        if xxs.shape[0] > memory_var:
            cutoff = int(xxs.shape[0] / 2)
            return np.concatenate((
                fftrec(xxs[0 : cutoff]),
                fftrec(xxs[cutoff :  ])
            ), axis = 0)
        else:
            return np.fft.fft(xxs, fft_n)
        
    fftxs = np.abs(
        fftrec(fft_in)[ : , lo : hi]
    )
    return fftxs, frqs[lo : hi]

# Fourier Power Feature Vector
# Equivalent to alt_ffv( < args > )[0] ** 2, ffv( < args > )[1] in output
def alt_fpfv(
    xs,                     # Input values: 2D numpy array
    Fs = 16000.,            # Sampling frequency
    epoch_size = 2 * 11,    # Epoch size
    fft_n = 2 ** 14,        # Increase results in greater dimensionality output
    lo = 12,                # Lowest frequency to include in output (incl.)
    hi = 30,                # Highest frequency to include in output (excl.)
    memory_var = 2 ** 14
):
    v, f = alt_ffv(xs, Fs, epoch_size, fft_n, lo, hi, memory_var)
    return v ** 2, f

# Spectral Entropy Vector
# Returns a vector of Spectral Entropy* for some input
# * Equivalent to asEntropy ( alt_fpfv ( < args > )[0] )
def alt_sev(
    xs,                     # Input values: 2D numpy array
    Fs = 16000.,            # Sampling frequency
    epoch_size = 2 * 11,    # Epoch size
    fft_n = 2 ** 14,        # Increase results in greater dimensionality output
    lo = 12,                # Lowest frequency to include in output (incl.)
    hi = 30,                # Highest frequency to include in output (excl.)
    memory_var = 2 ** 14
):
    return asEntropy(
        alt_fpfv(xs, Fs, epoch_size, fft_n, lo, hi)[0]
    )