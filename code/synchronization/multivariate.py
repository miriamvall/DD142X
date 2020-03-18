import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.stats import multivariate_normal as mvn

# multivariate normal distribution defined by mean and covariance matrix

# values of amplitude over time
mean = np.random.uniform(low=-0.0002, high=0.1, size=50)
# identity matrix -> all variables are independent
cov = np.eye(50)

# draws 25 samples from the distribution

n_samps_to_draw = 25

samples = mvn(mean, cov).rvs(n_samps_to_draw)

print(samples)