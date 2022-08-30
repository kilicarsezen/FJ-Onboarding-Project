from cProfile import label
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import STL


def test_stationarity(data):
     #Perform Dickey-Fuller test:    
    dftest = adfuller(data, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    return dfoutput


def outlier_analysis_plots(data, submodel, target, path="Outlier Analysis Mean"):

    # AD Fuller test and linear trend of the time series
    
            
    fig, axs = plt.subplots(3, 3, figsize=(25, 12))
    axs = axs.flat
    # original time series
    
    index=data.loc[data["Submodel"]==submodel, target].index
    data_plot_2 = data.loc[data["Submodel"]==submodel, "Orders_2"]
    ####--------------- STD=2---------------####
    axs[0].plot(data_plot_2, color='#0504aa', label='time series data')
    axs[0].hlines(y=data.loc[index, "uplim_2"], xmin=index[0], xmax=index[-1], linewidth=2, color='r', label="outlier upper limit")
    axs[0].hlines(y=data.loc[index, "lowlim_2"], xmin=index[0], xmax=index[-1],  linewidth=2, color='r', label="outlier lower limit")
    axs[0].set(xlabel="Date", ylabel="Nr of Orders", title=f"Number of Orders for {submodel} - std=2")
    axs[0].legend()
    # histogram of value distribution
    axs[3].hist(data_plot_2)
    axs[3].set(xlabel="Nr of Orders", ylabel="Count", title="Distribution")

    axs[6].boxplot(data_plot_2, vert=False, whis=0.75)
    axs[6].set(title="Boxplot of outlier corrected data")

    ####--------------- STD=2.5---------------####
    data_plot_2_5 = data.loc[data["Submodel"]==submodel, "Orders_2_5"]
    axs[1].plot(data_plot_2_5, color='#0504aa', label='time series data')
    axs[1].hlines(y=data.loc[index, "uplim_2_5"], xmin=index[0], xmax=index[-1], linewidth=2, color='r', label="outlier upper limit")
    axs[1].hlines(y=data.loc[index, "lowlim_2_5"], xmin=index[0], xmax=index[-1],  linewidth=2, color='r', label="outlier lower limit")
    axs[1].set(xlabel="Date", ylabel="Nr of Orders", 
               title=f"Number of Orders for {submodel} - std=2.5")
    axs[1].legend()

    axs[4].hist(data_plot_2_5)
    axs[4].set(xlabel="Date", ylabel="SEASONAL", title="Distribution")

    axs[7].boxplot(data_plot_2_5, vert=False, whis=0.75)
    axs[7].set(title="Boxplot of outlier corrected data")

    ####--------------- STD=3---------------####
    data_plot_3 = data.loc[data["Submodel"]==submodel, "Orders_3"]
    axs[2].plot(data_plot_3, color='#0504aa', label='time series data')
    axs[2].hlines(y=data.loc[index, "uplim_3"], xmin=index[0], xmax=index[-1], linewidth=2, color='r', label="outlier upper limit")
    axs[2].hlines(y=data.loc[index, "lowlim_3"], xmin=index[0], xmax=index[-1],  linewidth=2, color='r', label="outlier lower limit")
    axs[2].set(xlabel="Date", ylabel="Nr of Orders", 
               title=f"Number of Orders for {submodel} - std=3")
    axs[2].legend()

    axs[5].hist(data_plot_3)
    axs[5].set(xlabel="Date", ylabel="SEASONAL", title="Distribution")

    axs[8].boxplot(data_plot_3, vert=False, whis=0.75)
    axs[8].set(title="Boxplot of outlier corrected data")
    plt.tight_layout()

    fig.savefig('{}/{}_analysis.png'.format(path, submodel))
    plt.cla()
    plt.clf()

    plt.close()