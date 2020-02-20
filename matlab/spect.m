data = [gp_lfp1 gp_lfp2 gp_lfp17 gp_lfp18 gp_lfp3 gp_lfp20 gp_lfp21 gp_lfp22 gp_lfp23 ... 
    str_lfp11 str_lfp12 str_lfp13 str_lfp14 str_lfp15 str_lfp16 str_lfp26 str_lfp27 str_lfp28]
% Amount of data points to show
lim = 50
hold on
for i = 1:length(data)
    % Amount of data points
    n = 2^14;
    % Data
    y = data(i).values(1:n);
    % Sampling frequency
    Fs = 16000;
    % DFFT
    Y = fft(y, n);
    % Use complex conjugate to find reals
    Pyy = Y.*conj(Y)/n;
    % x-axis of plot
    f = Fs/n*(10:lim);
    % Action
    if i > length(data)/2
       plot(f, Pyy(11:lim + 1), "r")
    else
       plot(f, Pyy(11:lim + 1), "b")
    end
end