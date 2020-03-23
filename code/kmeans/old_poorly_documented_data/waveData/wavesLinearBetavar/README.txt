for f in range(12, 31):
allValues += genWaves(maxMean - f / 100, sigma * (maxMean - f / 100), channels, f, length, fs, phase=phase())

for f in np.concatenate((np.arange(1, 11), np.arange(31, 100))):
allValues += genWaves(maxMean - f / 100, 0.1 * (maxMean - f / 100), channels, f, length, fs, phase=phase())

# N(0, 1) noise to all data
allValues += 1 * np.random.normal(size = (channels, length * fs))