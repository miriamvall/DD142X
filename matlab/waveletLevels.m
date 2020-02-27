input = gp_lfp4.values(1:2^13);

wave = "db2";

[c, l] = wavedec(input, 6, wave);
approx = appcoef(c,l,wave);
[cd1,cd2,cd3,cd4,cd5,cd6] = detcoef(c,l,1:6);

nplots = 8;

subplot(nplots,1,7)
plot(approx)
title('Approximation Coefficients')
subplot(nplots,1,1)
plot(cd1)
title('Level 1 Detail Coefficients')
subplot(nplots,1,2)
plot(cd2)
title('Level 2 Detail Coefficients')
subplot(nplots,1,3)
plot(cd3)
title('Level 3 Detail Coefficients')
subplot(nplots,1,4)
plot(cd4)
title('Level 4 Detail Coefficients')
subplot(nplots,1,5)
plot(cd5)
title('Level 5 Detail Coefficients')
subplot(nplots,1,6)
plot(cd6)
title('Level 6 Detail Coefficients')
subplot(nplots,1,8)
plot(input)
title('values') 