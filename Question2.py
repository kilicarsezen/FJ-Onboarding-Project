from pickle import TRUE
from random import sample
from turtle import up
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from TS_Analysis import outlier_analysis_plots
from AutoArima import get_monthly_forcasts, plot_arima   


path="C:/Users/KilicarslanS/Documents/Onboarding test project/FJ-Onboarding-Project/"
sample_data = pd.read_excel("Monthly sample data.xlsx")
#sample_data.set_index("Month", inplace=True)
def replace_outlier(val, mean, std, dist):
    if val > mean + dist*std:
        return mean + dist*std 
    elif val < mean - dist*std:
        return mean - dist*std
    return val


def correct_outliers(data, target_name, dist, mean_=True):
    mean = data[target_name].mean()
    std_dev =  data[target_name].std(axis=0)
    up_lim = mean+dist*std_dev
    low_lim = mean-dist*std_dev
    if mean_:
        mean = data[target_name].mean()
        std_dev =  data[target_name].std(axis=0)
        Orders =  data[target_name].map(lambda x: replace_outlier(x, mean, std_dev, dist))
    else:
        median = data[target_name].median()
        std_dev =  data[target_name].std(axis=0)
        Orders =  data[target_name].map(lambda x: replace_outlier(x, median, std_dev, dist))
    return Orders, up_lim, low_lim

products = sample_data['Submodel'].unique()
sample_data_mean = sample_data.copy()
sample_data_median = sample_data.copy()
for product in products:
    index = sample_data.loc[sample_data['Submodel']==product].index
    
    sample_data_mean.loc[index, "Orders_2"], sample_data_mean.loc[index, "uplim_2"], sample_data_mean.loc[index, "lowlim_2"] = correct_outliers(data=sample_data.loc[index], target_name="Orders", dist=2)
    sample_data_mean.loc[index, "Orders_2_5"], sample_data_mean.loc[index, "uplim_2_5"], sample_data_mean.loc[index, "lowlim_2_5"] = correct_outliers(data=sample_data.loc[index], target_name="Orders", dist=2.5)
    sample_data_mean.loc[index, "Orders_3"], sample_data_mean.loc[index, "uplim_3"], sample_data_mean.loc[index, "lowlim_3"] = correct_outliers(data=sample_data.loc[sample_data['Submodel']==product], target_name="Orders", dist=3)
    
    sample_data_median.loc[index, "Orders_2"], sample_data_median.loc[index, "uplim_2"], sample_data_median.loc[index, "lowlim_2"] = correct_outliers(data=sample_data.loc[index], target_name="Orders", dist=2)
    sample_data_median.loc[index, "Orders_2_5"], sample_data_median.loc[index, "uplim_2_5"], sample_data_median.loc[index, "lowlim_2_5"] = correct_outliers(data=sample_data.loc[index], target_name="Orders", dist=2.5)
    sample_data_median.loc[index, "Orders_3"], sample_data_median.loc[index, "uplim_3"], sample_data_median.loc[index, "lowlim_3"] = correct_outliers(data=sample_data.loc[sample_data['Submodel']==product], target_name="Orders", dist=3, mean_=False)


    outlier_analysis_plots(sample_data_mean, submodel=product, target="Orders", path="Outlier Analysis Mean")
    outlier_analysis_plots(sample_data_median, submodel=product, target="Orders", path="Outlier Analysis Median")    

    sample_data.loc[index, "Orders"] = sample_data_median.loc[index, "Orders_2"]

list_of_forecasts_monthly = []
sample_data = sample_data.set_index('Month')

for product in products[0:1]:
    
    data_ = sample_data.loc[sample_data['Submodel']==product,'Orders']
    print(data_)
    fcast_range = pd.date_range(data_.index.max(), periods=9, freq="1M")
    fcast = get_monthly_forcasts(data_, m=12, seasonal=True)
    fcast_df = pd.DataFrame({"Submodel":product,"Forecast":fcast}, index=fcast_range)
    list_of_forecasts_monthly.append(fcast_df)
    sample_data_noOutlier_forecasts = pd.concat(list_of_forecasts_monthly)
    plot_arima(data_, fcast_df.loc[fcast_range,"Forecast"] )
    plt.show()

sample_data_noOutlier_forecasts.to_excel("Monthly_Forecast_9.xlsx")