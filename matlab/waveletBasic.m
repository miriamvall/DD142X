% Sample
input = gp_lfp4.values(1:2^13);

% Daubechies wavelet
wave = "db2";

% Decomposition vector c, bookkeeping vector l
% Don't understand em, I can't say
[c, l] = wavedec(input, 6, wave);
% Something
approx = appcoef(c,l,wave);

hold on

% Note that input plot is naïvely normalized
% Works well for 2^13, 2^14, 2^15 for the sake of
% visualizing what the transform approximation is doing
plot(mean(approx)/mean(input) * input);

% Stretch to sync up with input over time axis
plot((length(input)/length(approx)) * (0 : length(approx) - 1), approx, "black", "LineWidth", 1);

hold off
