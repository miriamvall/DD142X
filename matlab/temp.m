data = [gp_lfp20 gp_lfp21 gp_lfp22 gp_lfp23 str_lfp11 str_lfp12 str_lfp13 str_lfp14 stn_bua1]
hold on
for i = 1:9
    plot(data(i).values)
end

plot(stn_sua1.values)
