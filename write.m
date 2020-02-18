files = [str_lfp31 gp_lfp1 gp_lfp17 gp_lfp18 gp_lfp2 gp_lfp20 gp_lfp21 gp_lfp22 gp_lfp23 gp_lfp3 gp_lfp4 gp_lfp5 gp_lfp6 gp_lfp7 gp_lfp8 gp_lfp9 str_lfp11 str_lfp12 str_lfp13 str_lfp14 str_lfp15 str_lfp16 str_lfp26 str_lfp27 str_lfp28 str_lfp29 str_lfp30]
lim = 1584554
for i = 1:length(files)
    f = files(i)
    writematrix(transpose(f.values(1:500000)), "out/" + f.title + ".1.csv")
    writematrix(transpose(f.values(500001:1100000)), "out/" + f.title + ".2.csv")
    writematrix(transpsoe(f.values(1100001:lim)), "out/" + f.title + ".3.csv")
end
