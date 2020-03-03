import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sb 
#autocorrelation function
import statsmodels as sm 

# dataframe : 100 rows (1 row per second), 193 columns
dataframe = pd.read_csv(r"fourier_over_time_gp_lfp1.csv")

#print(dataframe.head())

#setting up vectors for sec = 0 and sec = 1
a = np.array(dataframe.loc[0,:])
v = np.array(dataframe.loc[1,:])
#discrete correlation of a and v
#result = np.correlate(a, v, "full")

# "full" mode returns results for each t where a and v have some overlap
#print(result)

#np.savetxt('result.csv', result, delimiter = ',')

sm.graphics.tsa.plot_acf(a,lags=v)
plt.show()

