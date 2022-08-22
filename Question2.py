from pickle import TRUE
from random import sample
import pandas as pd
sample_data = pd.read_excel("Sample Dataset.xlsx")
import numpy as np
from scipy import stats

def correct_outliers(data, target_name, mean=True, median=False, std=3):
    if mean:
        outliers = data[(np.abs(stats.zscore(data[target_name])) > std)].index
        data.loc[outliers, target_name] = data[target_name].mean()+std*data[target_name].std()
    if median:
        z_median = (data[target_name]-data[target_name].median())/data[target_name].std()
        outliers = data[(np.abs(z_median))>std].index
        data.loc[outliers, target_name] = data[target_name].mean()+std*data[target_name].std()
products = sample_data['Submodel'].unique()
sample_data_0 = sample_data.loc[sample_data['Submodel']==products[0],:]
