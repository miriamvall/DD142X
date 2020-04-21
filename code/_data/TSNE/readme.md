freqs, dataDict = load_all()
epochs = []

for session in sorted(dataDict.keys()):
    for channel in sorted(dataDict[session].keys()):
        for row in dataDict[session][channel]:
            epochs.append(row)
epochs = np.array(epochs)

loF = 5
hiF = 50
epochs = epochs[ : ,
    (freqs > loF) & (freqs < hiF)
]

embeddings = TSNE(n_components = 2).fit_transform(epochs.copy())

20:37 April 21