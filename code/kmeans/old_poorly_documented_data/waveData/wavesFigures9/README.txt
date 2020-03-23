# Relevant frequencies
for f in range(12, 31):
    allValues += genWaves(20, 5 * sigma, channels, f, length, fs, phase=phase())

# Loud, relatively chaotic noise
for f in range(2, 9):
    allValues += genWaves(10, sigma, channels, f, length, fs, phase=phase())
for f in range(35, 1000):
    allValues += genWaves(5, sigma, channels, f, length, fs, phase=phase())

# N(0, 10) noise to all data
allValues += 10 * np.random.normal(size = (channels, length * fs))