% Sampling frequency
Fs = 1000;
% Amount of data points
n = 2^13;
% Time steps are inverse of sampling frequency
x = 0:1/Fs:n;
% Our "data"
y = 10 * sin(5 * 2 * pi * x) + 7 * sin(12 * 2 * pi * x) + 15 * sin(27 * 2 * pi * x);
% How many frequencies to show
lim = 400;
% n-point DFFT
Y = fft(y, n);
% Take output of DFFT multiplied by its' complex conjugate
% Divide by input size(why?)
Pyy = Y.*conj(Y)/n;
% Plot x-axis
f = Fs/n*(0:lim);
plot(f, Pyy(1:lim + 1))