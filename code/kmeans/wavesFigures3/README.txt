# Relevant
for f in range(12, 31):
            # Interpret as AMPLITUDES sampled from N(10, sigma)
            # Everything else is meta
            allValues += genWaves(10, sigma, channels, f, length, fs)

# Loud, relatively chaotic noise
for f in range(2, 9):
    allValues += genWaves(20, 3 * sigma, channels, f, length, fs)
for f in range(35, 400):
    allValues += genWaves(20, 3 * sigma, channels, f, length, fs)