from os import walk, mkdir
import os, os.path
import numpy as np
import scipy.io as sio
import csv
import matplotlib.pyplot as plt
import math
import seaborn as sns 
import pandas as pd 
from sklearn.cluster import KMeans
from random import sample
from math import floor, pi
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
		countTrain = data[0][0:50] # take only one spike train (first one)
		rf = rateFunction(countTrain)
		lagged_rates = laggedSequence(rf,1)
		sc = autocorrelation(rf)
		firstOrder_returnMap = jointIntervalDensity(rf,lagged_rates)
		ffreqs, pwspec = powerSpectrum(data[0])
		p = pdf(data[0])
		se = spectral_entropy(p)
		print("spectral entropy = " + str(se))
		plt.plot(rf)
		fig = plt.gcf()
		fig.set_size_inches(18.5, 10.5)
		plt.title("Spiking Rate function")
		plt.xlabel("time window")
		plt.ylabel("spikes/second")
		plt.savefig("spikes/measurements/one_spike_train/rates.png")
		plt.clf()
		plt.hist(rf)
		plt.savefig("spikes/measurements/one_spike_train/ratesPDF.png")
		plt.clf()
		aux = (sns.jointplot(rf,lagged_rates,kind="hex")).set_axis_labels("rate function","rate function of lag 1")
		plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)  # shrink fig so cbar is visible
		cbar_ax = aux.fig.add_axes([.85, .25, .05, .4])
		plt.colorbar(cax=cbar_ax)
		fig = plt.gcf()
		fig.set_size_inches(13.5, 13.5)
		plt.savefig("spikes/measurements/one_spike_train/joint_rates.png")
		plt.clf()
		plt.plot(sc)
		plt.title("Serial Correlation Coefficients")
		plt.xlabel("time window")
		fig = plt.gcf()
		fig.set_size_inches(16.5, 13.5)
		plt.savefig("spikes/measurements/one_spike_train/serialcorr.png")
		plt.clf()
		plt.plot(ffreqs,pwspec)
		plt.title("Power Spectral Density")
		plt.xlabel("freqs")
		fig = plt.gcf()
		fig.set_size_inches(18.5, 10.5)
		plt.savefig("spikes/measurements/one_spike_train/pwspec.png")
		plt.clf()



def getInfoSpikesRegion(dataType):
	data = extractTrains("spikes/collected_data/"+dataType)
	ntrains = len(data)
	# measurements
	rates = [] 
	autocorr = []
	pwspec = []
	spec_entr = []
	for row in range(0,ntrains):
		# results 
		r = rateFunction(data[row][0:50])
		rates.append(r)
		autocorr.append(autocorrelation(r))
		aux = powerSpectrum(data[row][0:50])
		np.array(aux)
		pwspec.append(aux)
		spec_entr.append(spectral_entropy(data[row][0:50]))
	# correct format
	np.stack(rates)
	np.stack(autocorr)
	np.stack(pwspec)
	np.stack(spec_entr)
	# do plots and save them into png files
	for i in range(0,len(rates)):
		plt.plot(rates[i], alpha = 0.5)
	fig = plt.gcf()
	fig.set_size_inches(18.5, 10.5)
	plt.title("Spiking Rate function")
	plt.xlabel("time window")
	plt.ylabel("spikes/second")
	plt.savefig("spikes/measurements/"+ dataType[:-4] +"/spikingRates-" + dataType[:-4] +".png")
	plt.clf()
	for i in range(0,len(autocorr)):
		plt.plot(autocorr[i], alpha = 0.5)
	plt.title("Serial Correlation Coefficients")
	plt.xlabel("time window")
	fig = plt.gcf()
	fig.set_size_inches(16.5, 13.5)
	plt.savefig("spikes/measurements/"+ dataType[:-4] +"/autocorrelation-" + dataType[:-4] +".png")
	plt.clf()
	for i in range(0,len(pwspec)):
		plt.plot(pwspec[i][0],pwspec[i][1], alpha = 0.5)
	plt.title("Power Spectral Density")
	plt.xlabel("freqs")
	fig = plt.gcf()
	fig.set_size_inches(18.5, 10.5)
	plt.savefig("spikes/measurements/"+ dataType[:-4] +"/powerSpectrum-" + dataType[:-4] +".png")
	plt.clf()
	# plot histograms and scatter plot
	for i in range(0,ntrains):
		plt.hist(rates[i], alpha = 0.5)
		plt.title("Rate function probability distribution")
		plt.xlabel("Spiking rate")
		plt.ylabel("Counts")
	plt.savefig("spikes/measurements/"+ dataType[:-4] +"/ratesPDF-" + dataType[:-4] +".png")
	plt.clf()
	for i in range(0,ntrains):
		lagged_rates = laggedSequence(rates[i],1)
		aux = (sns.jointplot(rates[i],lagged_rates,kind="hex")).set_axis_labels("rate function","rate function of lag 1")
	plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)  # shrink fig so cbar is visible
	cbar_ax = aux.fig.add_axes([.85, .25, .05, .4])
	plt.colorbar(cax=cbar_ax)
	fig = plt.gcf()
	fig.set_size_inches(13.5, 13.5)
	plt.savefig("spikes/measurements/"+ dataType[:-4] +"/jointDistr-" + dataType[:-4] +".png")
	plt.clf()
	# another type of scatter plot
	for i in range(0,ntrains):
		lagged_rates = laggedSequence(rates[i],1)
		plt.scatter(rates[i],lagged_rates)
	plt.title("Joint rate probability distribution")
	plt.xlabel("Rate function")
	plt.ylabel("Rate function with time lag of 1")
	fig = plt.gcf()
	fig.set_size_inches(13.5, 13.5)
	plt.savefig("spikes/measurements/"+ dataType[:-4] +"/jointDistr2-" + dataType[:-4] +".png")
	plt.clf()
	# spectral entropies
	trains = np.arange(ntrains)
	for i in range(0,ntrains):
		plt.scatter(trains[i], spec_entr[i], alpha = 0.5)
		plt.savefig("spikes/measurements/"+ dataType[:-4] +"/specEntr-" + dataType[:-4] +".png")
	plt.clf()

	# attempt to plot a measurement for all channels

	# rate functions
	fig = plt.figure(1, figsize=(5, 3))

	ax = plt.gca()
	im = plt.imshow(rates, cmap = 'gray')
	if dataType[9:11] == "gp":
		plt.title("Rate function for 50 time windows for " + dataType[9:11] + " channels \n time window size = 0.5 seconds, rates in spikes/second")
	else:
		plt.title("Rate function for 50 time windows for " + dataType[9:12] + " channels \n time window size = 0.5 seconds, rates in spikes/second")
	plt.xlabel("time window")
	plt.ylabel("Channel")
	plt.yticks(np.arange(0,len(autocorr)))
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="3%", pad=0.25)
	plt.colorbar(im, cax=cax)
	plt.savefig("spikes/measurements/"+ dataType[:-4] + "/rates50tw")
	plt.clf()
	# rate functions
	fig = plt.figure(1, figsize=(5, 3))

	ax = plt.gca()
	im = plt.imshow(autocorr, cmap = 'gray')
	if dataType[9:11] == "gp":
		plt.title("Serial correlation coefficients for 50 time windows for " + dataType[9:11] + " channels \n time window size = 0.5 seconds")
	else:
		plt.title("Serial correlation coefficients for 50 time windows for " + dataType[9:12] + " channels \n time window size = 0.5 seconds")
	plt.xlabel("time window")
	plt.ylabel("Channel")
	plt.yticks(np.arange(0,len(rates)))
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="3%", pad=0.25)
	plt.colorbar(im, cax=cax)
	plt.savefig("spikes/measurements/"+ dataType[:-4] + "/autocorr50tw")
	plt.clf()

	return rates, autocorr, pwspec, spec_entr

# statistics about spikes of all available regions (gp,stn)
def getInfoAllSpikes():
	sr_gp, ac_gp, ps_gp, se_gp = getInfoSpikesRegion("sp_count_gp_sua.csv")
	sr_stn, ac_stn, ps_stn, se_stn = getInfoSpikesRegion("sp_count_stn_sua.csv")
	#plot the regions together
	for i in range(0,len(sr_gp)):
		plt.plot(sr_gp[i], 'r', alpha = 0.25)
	for i in range(0,len(sr_stn)):
		plt.plot(sr_stn[i], 'b', alpha = 0.25)
	fig = plt.gcf()
	fig.set_size_inches(18.5, 10.5)
	plt.title("Spiking Rate function")
	plt.xlabel("time window")
	plt.ylabel("spikes/second")
	plt.savefig("spikes/measurements/comp_gp_stn/spikingRates-gp_stn.png")
	plt.clf()

	for i in range(0,len(ac_gp)):
		plt.plot(ac_gp[i], 'r', alpha = 0.25)
	for i in range(0,len(ac_stn)):
		plt.plot(ac_stn[i], 'b', alpha = 0.25)
	plt.title("Serial Correlation Coefficients")
	plt.xlabel("time window")
	fig = plt.gcf()
	fig.set_size_inches(16.5, 13.5)
	plt.savefig("spikes/measurements/comp_gp_stn/autocorrelation-gp_stn.png")
	plt.clf()

	for i in range(0,len(ps_gp)):
		plt.plot(ps_gp[i][0],ps_gp[i][1], 'r', alpha = 0.25)
	for i in range(0,len(ps_stn)):
		plt.plot(ps_stn[i][0],ps_stn[i][1], 'b', alpha = 0.25)
	plt.title("Power Spectral Density")
	plt.xlabel("freqs")
	fig = plt.gcf()
	fig.set_size_inches(18.5, 10.5)
	plt.savefig("spikes/measurements/comp_gp_stn/powerSpectrum-gp_stn.png")
	plt.clf()

	for i in range(0,len(sr_gp)):
		plt.hist(sr_gp[i], alpha = 0.5,color='r')
	for i in range(0,len(sr_stn)):
		plt.hist(sr_stn[i], alpha = 0.5,color='b')
	plt.title("Rate function probability distribution")
	plt.xlabel("Spiking rate")
	plt.ylabel("Counts")
	plt.savefig("spikes/measurements/comp_gp_stn/ratesPDF-gp_stn.png")
	plt.clf()

	trains_gp = np.arange(len(se_gp))
	trains_stn = np.arange(len(se_stn))
	for i in range(0,len(se_gp)):
		plt.scatter(trains_gp[i], se_gp[i], alpha = 0.5,color='r')
	for i in range(0,len(se_stn)):
		plt.scatter(trains_stn[i], se_stn[i], alpha = 0.5,color='b')
	plt.savefig("spikes/measurements/comp_gp_stn/specEntr-gp_stn.png")
	plt.clf()

	for i in range(0,len(sr_gp)):
		lagged_rates = laggedSequence(sr_gp[i],1)
		sns.jointplot(sr_gp[i],lagged_rates,kind="hex", color='r')
	for i in range(0,len(sr_stn)):
		lagged_rates = laggedSequence(sr_stn[i],1)
		aux = (sns.jointplot(sr_stn[i],lagged_rates,kind="hex", color='b')).set_axis_labels("rate function","rate function of lag 1")
	plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)  # shrink fig so cbar is visible
	cbar_ax = aux.fig.add_axes([.85, .25, .05, .4])
	plt.colorbar(cax=cbar_ax)
	fig = plt.gcf()
	fig.set_size_inches(13.5, 13.5)
	plt.savefig("spikes/measurements/comp_gp_stn/jointProb-gp_stn.png")
	plt.clf()

	for i in range(0,len(sr_gp)):
		lagged_rates = laggedSequence(sr_gp[i],1)
		plt.scatter(sr_gp[i],lagged_rates, color='r')
	for i in range(0,len(sr_stn)):
		lagged_rates = laggedSequence(sr_stn[i],1)
		plt.scatter(sr_stn[i],lagged_rates, color='b')
	plt.title("Joint rate probability distribution")
	plt.xlabel("Rate function")
	plt.ylabel("Rate function with time lag of 1")
	fig = plt.gcf()
	fig.set_size_inches(13.5, 13.5)
	plt.savefig("spikes/measurements/comp_gp_stn/jointProb2-gp_stn.png")
	plt.clf()

	fig = plt.figure(1, figsize=(5, 3))

	# plot rates together
	all_r = np.concatenate((sr_gp,sr_stn))
	ax = plt.gca()
	im = plt.imshow(all_r, cmap = 'gray')
	plt.title("Rate function for 50 time windows for GP and STN channels \n time window size = 0.5 seconds, rates in spikes/second")
	plt.xlabel("time window")
	plt.ylabel("Channel")
	plt.yticks(np.arange(0,len(all_r)))
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="3%", pad=0.25)
	plt.colorbar(im, cax=cax)
	plt.savefig("spikes/measurements/comp_gp_stn/rates50tw")
	plt.clf()
	# plot autocorrelations together
	all_a = np.concatenate((ac_gp,ac_stn))
	ax = plt.gca()
	im = plt.imshow(all_a, cmap = 'gray')
	plt.title("Serial correlation coefficients for 50 time windows for GP and STN channels \n time window size = 0.5 seconds")
	plt.xlabel("time window")
	plt.ylabel("Channel")
	plt.yticks(np.arange(0,len(all_a)))
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="3%", pad=0.25)
	plt.colorbar(im, cax=cax)
	plt.savefig("spikes/measurements/comp_gp_stn/autocorr50tw")
	plt.clf()

	all_se = np.concatenate((se_gp,se_stn))
	np.savetxt("spikes/measurements/spec_entr.csv", all_se, delimiter=',', fmt='%f')
	
	

def test():
	#extractAllTypes()
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


# returns the interspike intervals of each time window (in ms)
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
	return result


# power spectrum (= power spectral density or noise spectrum) of spike train 
def powerSpectrum(spikeCountTrain):
	nWindows = len(spikeCountTrain)
	rates = rateFunction(spikeCountTrain)
	autocorr = autocorrelation(rates)
	# fft of autocorrelation
	fourier = np.fft.fft(autocorr)
	# amplitude spectrum = np.abs(fourier)
	pwSpectrum = np.abs(fourier)**2
	# frequencies associated to each fourier component
	freqs = np.fft.fftfreq(spikeCountTrain.size)
	idx = np.argsort(freqs)
	n_idx = len(idx)
	idx = idx[floor(n_idx/2):]
	return freqs[idx], pwSpectrum[idx]

# normalizes the power spectrum of a spike train to be visualized as a PDF
def pdf(spikeCountTrain):
	ff, ps = powerSpectrum(spikeCountTrain)
	nWindows = len(ps)
	pdf = np.zeros((nWindows))
	for i in range(0,nWindows):
		pdf[i] = ps[i] / np.sum(ps)
	return pdf

# returns spectral entropy of the probability density distribution
# output normalized to [0, 1]
def spectral_entropy(spikeCountTrain):
	p = pdf(spikeCountTrain)
	n = len(p)
	entr = np.zeros((n))
	for i in range(0,n):
		entr[i] = p[i] * np.log2(p[i])
	return - np.sum(entr)/ np.log2(n)

#-------------------STATISTICS-------------------

# input = power spectrum vectors of each channel
# returns vectors with the means of the channels over the time windows
def getMeansPw(data):
	#(nchannels,2,197 values)
	#correct format: (nchannels,197 values)
	aux = np.zeros((len(data),197))
	means = np.zeros((len(data),197))
	for i in range(0, len(data)):
		aux[i] = data[i][1]
		means[i] = aux[i].mean()
	return means

def getStdDevsPw(data):
	#(nchannels,2,197 values)
	#correct format: (nchannels,197 values)
	aux = np.zeros((len(data),197))
	stds = np.zeros((len(data),197))
	for i in range(0, len(data)):
		aux[i] = data[i][1]
		stds[i] = aux[i].std()
	return stds

def getMeans(data):
	means = np.zeros((len(data),2))
	for i in range(0,len(data)):
		means[i][0] = data[i].mean()
		means[i][1] = data[i].mean()
	return means 

def getStdDevs(data):
	stds = np.zeros((len(data),2))
	for i in range(0,len(data)):
		stds[i][0] = data[i].std()
		stds[i][1] = data[i].std()
	return stds

# to ease the plotting of the vectors
# plots means, std deviations, ratio of means/std devs
def plot_colors(type,data_gp, data_stn):
	plt.clf()
	
	if type == "pwspec":
		mean_gp = getMeansPw(data_gp)
		mean_stn = getMeansPw(data_stn)
		std_gp = getStdDevsPw(data_gp)
		std_stn = getStdDevsPw(data_stn)
	else:
		mean_gp = getMeans(data_gp)
		mean_stn = getMeans(data_stn)
		std_gp = getStdDevs(data_gp)
		std_stn = getStdDevs(data_stn)

	mean_all = np.concatenate((mean_gp,mean_stn))
	std_all = np.concatenate((std_gp,std_stn))

	matrices = [
		mean_gp, std_gp, std_gp / mean_gp,
		mean_stn, std_stn, std_stn / mean_stn,
		mean_all, std_all, std_all / mean_all
	]

	types = [
	    (channel_type, measurement)
	    for channel_type in ["gp", "stn", "all"]
	    for measurement  in ["mean", "st. dev", "st. dev : mean"]
	]

	for matrix, (channel_type, measurement), index in zip(
	    matrices,
	    types,
	    range(1, 10)
	):
		plt.rcParams["figure.figsize"] = (20, 10)
		plt.subplot(5, 2, index) # plots displayed in a single image
		# transpose to go from vertical to horizontal 
		plt.imshow(matrix.transpose(), extent=[0, len(matrix), 0, 1]) # display data as image
		plt.colorbar() # add colorbar to the plot
		plt.title(str(measurement) + " of " + type + " of " + str(channel_type))
		plt.xlabel("channels")
		plt.yticks([])

	fig = plt.gcf()
	fig.set_size_inches(13.5, 13.5)
	plt.savefig("spikes/measurements/statistics/" + type + ".png")
	



#---------------------MAIN---------------------------------

test()