1.
Feature vectors of fourier transform.
TSNE on feature vectors.
Each color/shape is an epoch. Similar colors/shapes do not necessarily mean anything.
Similar shapes are closer to each other in time.
Strong points are means for that epoch.
Ellipse is standard deviation.
Weak points are individual samples.

2.
As 1., fewer epochs.

3.
Fourier feature vector, followed by 2-component TSNE.
Embeddings clustered with K-means, 8 clusters.
Only first 100 epochs shown.
Second figure is cluster centers.

4.
As 3., 3-component TSNE.

5.
Fourier feature vector, followed by 4-component PCA.
Components have explained variance ratio sum 0.92.
K-means with 8 clusters.

6.
As 5., but each dimension in input samples scaled by ( * ratio)
	where <ratio> is the ratio of variance explained by that dimension's corresponding principal component.
4-cluster K-means.

7.
As 5., but with 4-cluster K-means.

8.
As 6., 8-cluster K-means.
