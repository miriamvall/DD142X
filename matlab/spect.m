data = [gp_lfp20 gp_lfp21 gp_lfp22 gp_lfp23 str_lfp11 str_lfp12 str_lfp13 str_lfp14 stn_bua1]
hold on
for i = 1:9
    y = data(i).values;
    Fs = 16000;
    step = 1/Fs;
    t = 0:step:100;
    n = 2^14;
    Y = fft(y, n);
    Pyy = Y.*conj(Y)/n;
    f = Fs/n*(0:40);
    plot(f, Pyy(1:41))
end

y = stn_sua1.values;
Fs = 16000;
step = 1/Fs;
t = 0:step:100;
n = 2^14;
Y = fft(y, n);
Pyy = Y.*conj(Y)/n;
f = Fs/n*(0:40);
plot(f, Pyy(1:41))
