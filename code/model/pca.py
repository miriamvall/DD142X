import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matlab_functions import getMatlabValues
from sklearn.decomposition import PCA

#####
#
#   It might be more relevant to use PCA on something that isn't raw data.
#   For example, 
#       epoch -> dft -> fourier/beta-feature vector
#       fourier/beta-feature vector -> PCA -> [Some sort of description of states epochs can be found in]
#   Preferably with shorter epochs than we've usually been working with.
#
#####

allChannels, _ = getMatlabValues("../_data/matlabData/NPR-075.b11.mat")

vals = np.array([
    v for k,v in allChannels.items()
    if "gp_lfp" in k or "str_lfp" in k
])[ : , 0:193*2**13].reshape((-1, 2 ** 10))

pca = PCA(n_components=3).fit(vals.copy())
trns = pca.transform(vals.copy())

aprx = np.array([
    np.array([ r * c for r, c in zip(row, pca.components_) ]).sum(axis = 0)
for row in trns])

vals_norm = np.linalg.norm(vals, axis = 1)
print("### Values ###")
print("Mean norm " + str(vals_norm.mean()))
print("Std norm " + str(vals_norm.std()))

aprx_norm = np.linalg.norm(aprx, axis = 1)
print("### Approximation by PCA ###")
print("Mean norm " + str(aprx_norm.mean()))
print("Std norm " + str(aprx_norm.std()))

diff_norm = np.linalg.norm(vals - aprx, axis = 1)
print("### Values - Approximation by PCA ###")
print("Mean norm " + str(diff_norm.mean()))
print("Std norm " + str(diff_norm.std()))

print("### Explained variance ratio sum ###")
print(pca.explained_variance_ratio_.sum())
print("### Explained variance per component ###")
print(pca.explained_variance_ratio_)

dim3 = True

if dim3:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(
        xs = [x for x,_,_ in trns], 
        ys = [y for _,y,_ in trns],
        zs = [z for _,_,z in trns],
        alpha = 0.01)
    plt.show()
else:
    plt.plot([x for x,_ in trns], [y for _,y in trns], '.', alpha = 0.01)
    plt.show()

