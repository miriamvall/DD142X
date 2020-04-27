from os import walk, mkdir
import os, os.path
import numpy as np
import scipy.io as sio
import csv
import matplotlib.pyplot as plt
import math
import seaborn as sns 
import pandas as pd 

#---------------OBTAINING DATA---------------------------

#returns spiking rates as ints
def extractTrains(inputFile):
	data = np.genfromtxt(inputFile, delimiter = ',')
	return data

# extracts all data of channels of the same type and stores in numpy 2D array 
# each channel gets a row
def extractType(path, dataType):
	data = []
	for(dirpath,dirnames,filenames) in walk(path):
		for dirname in dirnames:
			print(dirname)
			for(_,_, filenames) in walk(path+'/'+dirname):
				for filename in filenames:
					if filename == dataType:
						aux = extractTrains(path+'/'+dirname+'/'+filename)
						if aux.size != 0: #check if file not empty
							ndim = aux.ndim #more than a row (spike train)
							if ndim > 1:
								for i in range(0,len(aux)):
									data.append(aux[i, 0:393])
							else:
								aux = aux[0:393] #all trains with same length
								data.append(aux)
	np.stack(data)
	np.savetxt("spikes/collected_data/" + dataType[:-4] + ".csv", data, delimiter=',', fmt='%f')
	return data

def extractionByType(dataType):
	inputDir = "../_data/csvData/summary"
	return extractType(inputDir,dataType)

def extractAllTypes():
	gp_sp = extractionByType("sp_count_gp_sua.csv")
	stn_gp = extractionByType("sp_count_stn_sua.csv")
	gp_spect = extractionByType("spect_ent_gp.csv")
	str_spect = extractionByType("spect_ent_str.csv")

#---------------------------GETTING MEASUREMENTS---------------------------

# get plots for spectral entropies of the regions (gp,str)
def getInfoSpectEntropy():
	data_gp = extractTrains("spikes/collected_data/spect_ent_gp.csv")
	data_str = extractTrains("spikes/collected_data/spect_ent_str.csv")
	ntrains_gp = len(data_gp)
	ntrains_str = len(data_str)
	for i in range(0,ntrains_gp):
		plt.plot(data_gp[i], 'r', alpha = 0.5)
	for i in range(0,ntrains_str):
		plt.plot(data_str[i], 'b', alpha = 0.5)
	plt.savefig("spikes/measurements/entropy_gp_str/entropy_gp_str.png")
	plt.clf()

def getInfo1Spike():
		data = extractTrains("spikes/collected_data/sp_count_gp_sua.csv")
		ntrains = len(data)
		countTrain = data[0] # take only one spike train (first one)
		rf = rateFunction(countTrain)
		isi = ISI(countTrain)
		lagged_isi = laggedSequence(isi,1)
		sc = autocorrelation(isi)
		firstOrder_returnMap = jointIntervalDensity(isi,lagged_isi)
		ffreqs, pwspec = powerSpectrum(data[0])
		plt.plot(isi)
		plt.savefig("spikes/measurements/one_train/isi.png")
		plt.clf()
		print(len(isi))
		plt.hist(isi)
		plt.title("ISI Probability Distribution")
		plt.xlabel("ISI (seconds)")
		plt.ylabel("counts")
		plt.savefig("spikes/measurements/one_train/isiPDF.png")
		plt.clf()
		sns.jointplot(isi,lagged_isi,kind="hex")
		plt.savefig("spikes/measurements/one_train/joint_isi.png")
		plt.clf()
		plt.plot(sc)
		plt.savefig("spikes/measurements/one_train/serialcorr.png")
		plt.clf()
		plt.plot(ffreqs,pwspec)
		plt.savefig("spikes/measurements/one_train/pwspec.png")

def getInfoSpikesRegion(dataType):
	data = extractTrains("spikes/collected_data/"+dataType)
	ntrains = len(data)
	# measurements
	rates = [] 
	isis = []
	autocorr = []
	pwspec = []
	for row in range(0,ntrains):
		# results 
		rates.append(rateFunction(data[row]))
		isi = ISI(data[row])
		isis.append(isi)
		autocorr.append(autocorrelation(isi))
		aux = powerSpectrum(data[row])
		np.array(aux)
		pwspec.append(aux)
	# correct format
	np.stack(rates)
	np.stack(isis)
	np.stack(autocorr)
	np.stack(pwspec)
	# save into csv files
	np.savetxt("spikes/measurements/"+ dataType[:-4] +"/spikingRates.csv", rates, delimiter = ',', fmt = "%f")
	np.savetxt("spikes/measurements/"+ dataType[:-4] +"/isis.csv", isis, delimiter = ',', fmt = "%f")
	np.savetxt("spikes/measurements/"+ dataType[:-4] +"/autocorrelation.csv", autocorr, delimiter = ',', fmt = "%f")
	# do plots and save them into png files
	for i in range(0,len(rates)):
		plt.plot(rates[i], alpha = 0.5)
	plt.savefig("spikes/measurements/"+ dataType[:-4] +"/spikingRates-" + dataType[:-4] +".png")
	plt.clf()
	for i in range(0,len(isis)):
		plt.plot(isis[i], alpha = 0.5)
	plt.savefig("spikes/measurements/"+ dataType[:-4] +"/isis-" + dataType[:-4] +".png")
	plt.clf()
	for i in range(0,len(autocorr)):
		plt.plot(autocorr[i], alpha = 0.5)
	plt.savefig("spikes/measurements/"+ dataType[:-4] +"/autocorrelation-" + dataType[:-4] +".png")
	plt.clf()
	for i in range(0,len(pwspec)):
		plt.plot(pwspec[i][0],pwspec[i][1], alpha = 0.5)
	plt.savefig("spikes/measurements/"+ dataType[:-4] +"/powerSpectrum-" + dataType[:-4] +".png")
	plt.clf()
	# plot histograms and scatter plot
	for i in range(0,ntrains):
		plt.hist(isis[i], alpha = 0.5)
	plt.savefig("spikes/measurements/"+ dataType[:-4] +"/isiPDF-" + dataType[:-4] +".png")
	plt.clf()
	for i in range(0,ntrains):
		lagged_isi = laggedSequence(isis[i],1)
		sns.jointplot(isis[i],lagged_isi,kind="hex")
	plt.savefig("spikes/measurements/"+ dataType[:-4] +"/jointDistr-" + dataType[:-4] +".png")
	plt.clf()
	return rates, isis, autocorr, pwspec

# statistics about spikes of all available regions (gp,stn)
def getInfoAllSpikes():
	sr_gp, isi_gp, ac_gp, ps_gp = getInfoSpikesRegion("sp_count_gp_sua.csv")
	sr_stn, isi_stn, ac_stn, ps_stn = getInfoSpikesRegion("sp_count_stn_sua.csv")
	#plot the regions together
	for i in range(0,len(sr_gp)):
		plt.plot(sr_gp[i], 'r', alpha = 0.25)
	for i in range(0,len(sr_stn)):
		plt.plot(sr_stn[i], 'b', alpha = 0.25)
	plt.savefig("spikes/measurements/comp_gp_stn/spikingRates-gp_stn.png")
	plt.clf()
	
	for i in range(0,len(isi_gp)):
		plt.plot(isi_gp[i], 'r', alpha = 0.25)
	for i in range(0,len(isi_stn)):
		plt.plot(isi_stn[i], 'b', alpha = 0.25)
	plt.savefig("spikes/measurements/comp_gp_stn/intervalDistr-gp_stn.png")
	plt.clf()

	for i in range(0,len(ac_gp)):
		plt.plot(ac_gp[i], 'r', alpha = 0.25)
	for i in range(0,len(ac_stn)):
		plt.plot(ac_stn[i], 'b', alpha = 0.25)
	plt.savefig("spikes/measurements/comp_gp_stn/autocorrelation-gp_stn.png")
	plt.clf()

	for i in range(0,len(ps_gp)):
		plt.plot(ps_gp[i][0],ps_gp[i][1], 'r', alpha = 0.25)
	for i in range(0,len(ps_stn)):
		plt.plot(ps_stn[i][0],ps_stn[i][1], 'b', alpha = 0.25)
	plt.savefig("spikes/measurements/comp_gp_stn/powerSpectrum-gp_stn.png")
	plt.clf()

	for i in range(0,len(isi_gp)):
		plt.hist(isi_gp[i], alpha = 0.5,color='r')
	for i in range(0,len(isi_stn)):
		plt.hist(isi_stn[i], alpha = 0.5,color='b')
	plt.savefig("spikes/measurements/comp_gp_stn/isiPDF-gp_stn.png")
	plt.clf()

	for i in range(0,len(isi_gp)):
		lagged_isi = laggedSequence(isi_gp[i],1)
		sns.jointplot(isi_gp[i],lagged_isi,kind="hex", color='r')
	for i in range(0,len(isi_stn)):
		lagged_isi = laggedSequence(isi_stn[i],1)
		sns.jointplot(isi_stn[i],lagged_isi,kind="hex", color='b')
	plt.savefig("spikes/measurements/comp_gp_stn/jointProb-gp_stn.png")
	plt.clf()


def test():
	extractAllTypes()
	#getInfo1Spike()
	#getInfoSpikesRegion("sp_count_gp_sua.csv")
	getInfoAllSpikes()
	
		

#--------------------CALCULATION OF MEASURES--------------------


# poisson rate function (in spikes/s)
def rateFunction(spikeCountTrain):
	nWindows = len(spikeCountTrain)
	time = 0.5 # 500 ms 
	asRates = np.zeros((nWindows))
	for i in range(0,nWindows):
		asRates[i] = spikeCountTrain[i] / time
	return asRates


# returns the interspike intervals of each time window
def ISI(spikeCountTrain):
	nWindows = len(spikeCountTrain)
	time = 0.005
	asIntervals = np.zeros((nWindows))
	for i in range(0,nWindows):
		if spikeCountTrain[i] == 0:
			asIntervals[i] = 0.005
		else:
			asIntervals[i] = time / spikeCountTrain[i]
	return asIntervals



#returns the input sequence + time-lagged input sequence (with chosen number of lags)
def laggedSequence(intervals, lags):
	return np.roll(intervals,lags)
		

# returns plot of joint interval density from same or different spike train, along with a histogram
def jointIntervalDensity(isi1, isi2):
	nintervals = len(isi1)
	#pairs = np.zeros((nintervals))
	pairs = []
	for i in range(0,nintervals):
		#pairs[i] = (isi1[i], isi2[i])
		pairs.append((isi1[i], isi2[i]))
	np.array(pairs)
	return pairs

# correlation coefficient between two time-lagged sequences
def corrcoefficient(isi, lags):
	lagged = laggedSequence(isi,lags)
	seqs = [isi,lagged]
	np.array(seqs)
	corrMatrix = np.corrcoef(seqs)
	print(corrMatrix)
	return corrMatrix[0,1]


#  Normalized autocorrelation function 
def autocorrelation(intervalDistr):
	nWindows = len(intervalDistr)
	# the autocorrelation is a sum of interspike intervals convolved with itself
	result = np.array([1]+[np.corrcoef(intervalDistr[:-i], intervalDistr[i:])[0,1] \
		for i in range(1, nWindows)])
	#fixing nan values 
	if np.isnan(result).any():
		pos = np.argwhere(np.isnan(result))
		#replace
		for i in range(0,len(pos)):
			result[pos] = result[pos-1]
	#from last element
	result[nWindows-1] = 1
	#print(result)
	return result


# power spectrum (= power spectral density or noise spectrum) of spike train 
def powerSpectrum(spikeCountTrain):
	nWindows = len(spikeCountTrain)
	isi = ISI(spikeCountTrain)
	autocorr = autocorrelation(isi)
	# fft of autocorrelation
	fourier = np.fft.fft(autocorr)
	# amplitude spectrum = np.abs(fourier)
	pwSpectrum = np.abs(fourier)**2
	# frequencies associated to each fourier component
	time_step = 1/30
	freqs = np.fft.fftfreq(spikeCountTrain.size, time_step)
	idx = np.argsort(freqs)
	return freqs[idx], pwSpectrum[idx]

#---------------------MAIN---------------------------------

test()