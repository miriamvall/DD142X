import numpy as np

#   Returns a vector of abs(x(f)) for f in [12, 30]
#   x(f) is fourier transform of f over xs
def fft_feature_vector(xs, Fs = 16000., epoch_size = 2 ** 11, fft_n = 2 ** 14):

    # Pad with zeroes for more frequency outputs
    # Compare np.fft.fftfreq(n, 1/16000) for n = 2**11, 2**14
    fft_in = np.zeros((xs.shape[0], fft_n))
    fft_in[ : , 0:epoch_size] = xs

    frqs = np.fft.fftfreq(fft_n, 1./Fs)
    lo = np.where(frqs > 12)[0][0]
    hi = np.where(frqs > 30)[0][0]
    fftxs = np.abs(np.fft.fft(fft_in)[:,lo:hi])

    return fftxs, frqs[lo:hi]

def main():
    from matlab_functions import getMatlabValues
    import matplotlib.pyplot as plt

    print("I/O")
    # Extract raw data from files
    vals, _ = getMatlabValues("../_data/matlabData/NPR-075.b11.mat")
    vals = np.array([
        v for k, v in vals.items() if "str_lfp" in k or "gp_lfp" in k
    ])

    # "Constants"
    channels = vals.shape[0]
    Fs = 16000.
    dt = 1. / Fs
    epoch_size = 2**11
    epochs = int(vals[0].shape[0] / epoch_size)

    # Cropping
    vals = vals[ : , 0 : epoch_size * epochs]

    # To epochs
    asEpochs = vals.copy().reshape((-1, epoch_size))

    print("Feature vectors")
    # Create feature vectors
    # See documentation for assumptions of function
    featvs, frqs = fft_feature_vector(asEpochs, epoch_size=epoch_size, fft_n=2**15)
    
    # print("TSNE")
    # from sklearn.manifold import TSNE
    # embedded = TSNE(n_components=3).fit_transform(featvs.copy())

    print("PCA")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=4).fit(featvs.copy())
    embedded = pca.transform(featvs.copy())
    embedded = embedded * np.array(pca.explained_variance_ratio_)

    # embedded = embedded.reshape((channels, -1, 2))
    # for ep, col in zip(range(0, 6), [
    #     'bx', 'gx', 'rx', 'cx', 'mx', 'yx',
    #     # 'b.', 'g.', 'r.', 'c.', 'm.', 'y.',
    #     # 'bo', 'go', 'ro', 'co', 'mo', 'yo',
    #     # 'b+', 'g+', 'r+', 'c+', 'm+', 'y+',
    # ]):
    #     plt.plot(
    #         [x for x,_ in embedded[:,ep,:]],
    #         [y for _,y in embedded[:,ep,:]],
    #         col,
    #         alpha = 0.2
    #     )

    # for ep, col in zip(range(0, 6), [
    #     'bx', 'gx', 'rx', 'cx', 'mx', 'yx',
    #     # 'b.', 'g.', 'r.', 'c.', 'm.', 'y.',
    #     # 'bo', 'go', 'ro', 'co', 'mo', 'yo',
    #     # 'b+', 'g+', 'r+', 'c+', 'm+', 'y+',
    # ]):
    #     meanv = embedded[:,ep,:].mean(axis = 0)
    #     stadv = embedded[:,ep,:].std (axis = 0)
    #     plt.plot([meanv[0]], [meanv[1]], col)
    #     angles = np.linspace(-np.pi, np.pi, 60)
    #     plt.plot(
    #         [meanv[0] + np.cos(v) * stadv[0] for v in angles],
    #         [meanv[1] + np.sin(v) * stadv[1] for v in angles],
    #         col[0],
    #         alpha = 0.1
    #     )
    
    # plt.show()

    from sklearn.cluster import KMeans
    from random import sample
    training_n = int(embedded.shape[0] / 10)
    training_data = np.array([ sample(list(embedded), training_n) ])[0]

    print(training_data.shape)

    print("KMeans")
    kmeans = KMeans(n_clusters = 8).fit(training_data)
    predictions = kmeans.predict(embedded).reshape((channels, -1))
    plt.subplot(211)
    plt.imshow(predictions[:,0:100])
    plt.colorbar()
    plt.subplot(212)
    plt.imshow(kmeans.cluster_centers_)
    plt.colorbar()
    plt.show()

    # # print("Normalization")
    # # featvs = np.array([
    # #     (v - v.mean()) / v.std() for v in featvs
    # # ])

    # from sklearn.cluster import KMeans
    # from random import sample

    # print("K-Means")
    # training_n = int(featvs.shape[0] / 10)
    # training_data = np.array(
    #     sample(list(featvs), training_n)
    # )
    # kmeans = KMeans(n_clusters=4).fit(training_data)
    # predictions = kmeans.predict(featvs)
    # predictions = predictions.reshape((channels, -1))

    # plt.imshow(predictions[:,0:193])
    # plt.savefig("1.png")
    # plt.clf()
    # plt.imshow(predictions[:,193:2*193])
    # plt.savefig("2.png")
    # plt.clf()
    # plt.imshow(predictions[:,2*193:3*193])
    # plt.savefig("3.png")
    # plt.clf()
    # plt.imshow(predictions[:,3*193:775])
    # plt.savefig("4.png")
    # plt.clf()
    # plt.imshow(kmeans.cluster_centers_)
    # plt.colorbar()
    # plt.savefig("5.png")
    # plt.clf()

    # print("TSNE of cluster centers")
    # from sklearn.manifold import TSNE
    # embedded = TSNE().fit_transform(kmeans.cluster_centers_.copy())
    # plt.plot([x for x,_ in embedded], [y for _,y in embedded], 'x')
    # plt.savefig("6.png")

    # plt.close()

    # from sklearn.decomposition import PCA

    # print("PCA")
    # pca = PCA(n_components=4).fit(featvs.copy())
    # asPC = pca.transform(featvs.copy())
    # print("Explained variance ratio sum: " + str(pca.explained_variance_ratio_.sum()))

    # from sklearn.manifold import TSNE

    # print("TSNE")
    # tsne_embedded_pca = TSNE().fit_transform(asPC.copy())
    
    # print("Plotting")
    # plt.plot(
    #     [x for x,_ in tsne_embedded_pca], 
    #     [y for _,y in tsne_embedded_pca], 
    #     'x', alpha = 0.05)
    # plt.show()

#main()