% Measure prominent frequencies of FFT over time

% Example data
values = str_lfp15.values;
% n for fft; time measure
width = 2^13;
% Amount of intervals of width for which to measure fft
nMax = floor(length(values)/width);
% Results, fft-prominent frequencies over time
% Increments of width/FS (~0.5s) for time
fs = zeros(width, nMax);
% Sampling frequency
FS = 16000;
for n = 1:nMax
    y = values((n - 1) * width + 1 : (n * width) + 1);
    Y = fft(y, width);
    Pyy = Y.*conj(Y)/width;
    fs(:,n) = Pyy;
end

xs = (width/FS) * (1:nMax);
%hold on
%for i = 10:10

    %plot(xs, fs(i,:));
%end
%hold off

writematrix(fs(1:100,:), "fourier_over_time_str_lfp15.csv")

% Pick a frequency, do wavelet approximation of it
%ps = fs(20,:);
%wave = "db2";
%[c, l] = wavedec(ps, 3, wave);
%approx = appcoef(c,l,wave);

%hold on

%plot(xs, ps)
% Manually scaled
%plot(3.8 * (1:length(approx)), (approx) - 0.05)

%hold off

