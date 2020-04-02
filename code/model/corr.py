import numpy as np
import matplotlib.pyplot as plt

# input data is list of 2D matrices - each matrix's rows representing the frequencies of a channel, cols are epochs
# outputs a quadratic matrix with the correlation coefficient between every pair of channels
def corrcoef(fftValues):

	# vectorize the matrices to calculate the correlation coefficient 
	# we store the vectors in a matrix

	nchannels = len(fftValues)
	nFreqs = fftValues[0].shape[0]
	nEpochs = fftValues[0].shape[1]

	valuesMatrix = np.zeros((nchannels, nFreqs*nEpochs))

	for i in range(0,nchannels):
		valuesMatrix[i] = fftValues[i].ravel()

	CC = np.corrcoef(valuesMatrix)

	return CC







