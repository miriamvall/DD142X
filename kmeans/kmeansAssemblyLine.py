import pandas as pd 
import numpy as np 
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from mpl_toolkits.mplot3d import Axes3D
from os import walk

for (dirpath, dirnames, filenames) in os.walk("../fourierdata/"):
    print(filenames)