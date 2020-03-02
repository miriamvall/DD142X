import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sb 
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

# %matplotlib inline 

from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

#we import a csv file with the data from 10 samples 

dataframe = pd.read_csv(r"samples_1second.csv")

# JUST VISUALIZING SHIT FOR NOW
 
#to see a statistical information table provided 
#by pandas dataframe

#dataframe.describe()

#to know how many "members" we have for each brain region

#print(dataframe.groupby('br').size())

#to graphically visualize the density of
#our data to have an idea of its dispersion

#dataframe.drop(['br'],1).hist()
#plt.show()

#WE DEFINE THE INPUT (taking all the fourier features)

X = np.array(dataframe.loc[:,'f1':'f193'])
X.shape

# K - MEANS ALGORITHM

#there's a way to decide which value of K is best but I'm
#just going ahead with K = 4 (can be changed)

kmeans = KMeans(n_clusters=4).fit(X)
centroids = kmeans.cluster_centers_
#print(centroids)

#3D plot with colors for the groups to see if they are different
# (stars mark the center of a cluster) :

#predicting the clusters
labels = kmeans.predict(X)
# to see what samples belong to which cluster
print(labels)
#getting the cluster centers
C = kmeans.cluster_centers_
#one color per cluster
colors=['red','blue','green','yellow']
assign=[]
for row in labels:
    assign.append(colors[row])
 
fig = plt.figure()
ax = Axes3D(fig)
#plotting the clusters - ONLY WITH 3 DIMENSIONS (x,y,z axis), 
# it still works but we should find another way to plot the results
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c = assign, s=60)
#stars for the centroids
ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c=colors, s=1000)
plt.show()