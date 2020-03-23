# Relevant frequencies
for f in range(12, 31):
    allValues += genWaves(10, sigma, channels, f, length, fs, phase=phase())

# Loud, relatively chaotic noise
for f in range(2, 9):
    allValues += genWaves(10, sigma, channels, f, length, fs, phase=phase())
for f in range(35, 400):
    allValues += genWaves(10, sigma, channels, f, length, fs, phase=phase())

# N(0, 10000) noise to all data
allValues += 10000 * np.random.normal(size = (channels, length * fs))