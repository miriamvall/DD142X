channels = 25
length = 60
fs = 16000
nClasses = 8
phase = lambda: np.random.uniform(0, pi)

sigmas = [i / 2 for i in range(1, 21)]
done = 0
for sigma in sigmas:
    NOW = time.localtime()
    print(f"%s:%s:%s" % (NOW.tm_hour, NOW.tm_min, NOW.tm_sec))
    allValues = np.zeros((channels, length * fs))

    # Relevant frequencies
    for f in range(12, 31):
        allValues += genWaves(10, sigma, channels, f, length, fs, phase=phase())

    # Loud, relatively chaotic noise
    for f in range(2, 9):
        allValues += genWaves(20, 3 * sigma, channels, f, length, fs, phase=phase())
    for f in range(35, 400):
        allValues += genWaves(20, 3 * sigma, channels, f, length, fs, phase=phase())

    # N(0, 1) noise to all data
    # Small effect; mean(abs(data)) >>> 1 and std(abs(data)) >> mean(abs(data))
    allValues += np.random.normal(size = (channels, length * fs))