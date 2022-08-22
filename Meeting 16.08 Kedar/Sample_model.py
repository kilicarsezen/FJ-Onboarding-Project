from cgi import test
from random import sample
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_predict
sample_data = pd.read_excel("Sample Dataset.xlsx")
print(sample_data.dtypes)
products = sample_data['Submodel'].unique()
first_product = sample_data.loc[sample_data['Submodel']==products[0], ['Month', 'Orders']].copy()
first_product  = first_product.set_index('Month')
# plt.plot(sample_data.loc[sample_data['Submodel']==products[0],'Month'], sample_data.loc[sample_data['Submodel']==products[0], 'Orders'])
# plt.show()
def test_stationarity(timeseries):    
    #Determing rolling statistics
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()
    plt.plot(timeseries, color='blue',label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    #Perform Dickey-Fuller test:
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
test_stationarity(first_product)

print(first_product.Orders)
fig, axes = plt.subplots(3, 2, sharex=False)
axes[0, 0].plot(first_product); axes[0, 0].set_title('Original Series')
plot_acf(first_product, ax=axes[0, 1])

# 1st Differencing
axes[1, 0].plot(first_product['Orders'].diff()); axes[1, 0].set_title('1st Order Differencing')
plot_acf(first_product.diff().dropna(), ax=axes[1, 1])

# 2nd Differencing
axes[2, 0].plot(first_product['Orders'].diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
plot_acf(first_product.diff().diff().dropna(), ax=axes[2, 1])


#pacf to decide the AR term
fig, axes = plt.subplots(1, 2, sharex=False)
axes[0].plot(first_product.diff(2)); axes[0].set_title('2nd Differencing')
axes[1].set(ylim=(0,5))
plot_pacf(first_product.diff(2).dropna(), lags=10, ax=axes[1])


#acf to decide the MA term
fig, axes = plt.subplots(1, 2, sharex=False)
axes[0].plot(first_product.diff(2)); axes[0].set_title('2nd Differencing')
axes[1].set(ylim=(0,1.2))
plot_acf(first_product.diff(2).dropna(), ax=axes[1])


# 1,1,1 ARIMA Model
model = ARIMA(first_product, order=(0,2,0))
model_fit = model.fit()
print(model_fit.summary())

residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1,2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])

forecasts = model_fit.predict(type='levels')
forecasts_ = model_fit.predict(start='2019-01-01', end='2019-12-01')
fig, ax = plt.subplots()
# ax.plot(forecasts_, 'r', label='forecasts diff')
ax.plot(forecasts, color='green', label='forecasts')
ax.plot(first_product, 'b', label='original ts')
ax.set_title("forcasts vs original")
plt.legend(loc="upper right")
plt.show()  
