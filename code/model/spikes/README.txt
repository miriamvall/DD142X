
"spikes" folder with directories:

	- "collected_data" for the csv files with all the channels for each type of data (sua and spectral entropies): a row for 		  each channel (= spike count train), columns are time windows
	  *for sua, data available on GP and STN // for spectral entropy, data available on GP and STR

	- "measurements" containing folders with measurements of each type of data + comparisons between spiking activity of GP and STN


inside of measurement dir:

	- one_spike_train: measurements on one single spike count train (of GP)

	- sp_count_gp_sua: measurements on all spike count trains (recorded channels) of GP

	- sp_count_stn_sua: measurements on all spike count trains of STN

	- comp_gp_stn: comparisons between measurements of GP and STN

	- entropy_gp_str: (pending - right now, just a plot of the spectral entropies of both GP and STR)


MEASUREMENTS:

Possible relevant measures to see the degree of independence between the spikes over time, and how it varies (plots are still pending to be named)

	- Spiking rates of each time window

	- Inter-spike intervals (ISI) of each time window: time between each pair of spikes

	- estimation of the ISI probability density function, from the histogram of the ISI

	- joint probability of the ISI sequence and the same ISI sequence with a time lag of 1 second: to see the probability that each 	  interval follows another. An overall upward trend in the scatter diagram indicates positive correlations between successive 		  intervals; this loosely means that short intervals tend to be followed by short ones, and long intervals by long ones. Departures 		  from independence are reflected in the symmetries of the scatter diagram itself.

	- serial correlation coefficients (inside range [-1,1]) between the intervals = autocorrelation of ISI sequence. A sufficiently great 		  monotone increase or decrease in the firing rate over time will contribute a positive component to each serial correlation 		  coefficient, on lags of high order.

	- Power spectrum (spectral density, noise spectrum) of the autocorrelation function. It's the squared of the fourier transform of the 		  autocorrelation function. It shows at which frequencies the variations are strong and at which freqs they're weak.


