% Measure prominent frequencies of FFT over time
for vals = [  gp_lfp1, gp_lfp2, gp_lfp3, gp_lfp4, gp_lfp5, ...
              gp_lfp6, gp_lfp7, gp_lfp8, gp_lfp9, gp_lfp17, ...
              gp_lfp18, gp_lfp20, gp_lfp21, gp_lfp22, gp_lfp23 ...
              str_lfp11, str_lfp12, str_lfp13, str_lfp14 ...
              str_lfp15, str_lfp16, str_lfp26, str_lfp27 ... 
              str_lfp28, str_lfp29, str_lfp30, str_lfp31]
    % Example data
    %values = str_lfp15.values;
    % n for fft; time measure
    width = 2^13;
    % Amount of intervals of width for which to measure fft
    nMax = floor(length(vals.values)/width);
    % Results, fft-prominent frequencies over time
    % Increments of width/FS (~0.5s) for time
    fs = zeros(width, nMax);
    % Sampling frequency
    FS = 16000;
    for n = 1:nMax
        y = vals.values((n - 1) * width + 1 : (n * width) + 1);
        Y = fft(y, width);
        % Pyy = Y.*conj(Y)/width;
        Pyy = abs(Y);
        fs(:,n) = Pyy;
    end

    writematrix(fs(8:35,:), "../fourierdata/fourier_over_time_" + vals.title + ".csv")
end
