from cgi import print_environ, test
from dbm import ndbm
from operator import invert
import pandas as pd
from pandas.tseries.offsets import DateOffset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import math
from scipy import signal
from sklearn.preprocessing import StandardScaler
from functools import partial
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (TimeSeriesSplit, train_test_split, cross_val_score)
from sklearn.preprocessing import StandardScaler, scale
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, STATUS_FAIL
import xgboost as xgb
from statsmodels.tsa.stattools import pacf, acf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer
from sklearn.preprocessing import OneHotEncoder
from statsmodels.tsa.seasonal import STL

rng = np.random.default_rng()
sample_data = pd.read_excel("Sample Dataset.xlsx")
sample_data = sample_data.set_index('Month')

#Q1.a
products = sample_data['Submodel'].unique()
sample_data_q = []
sample_data_m = []

for product in products:
    df=sample_data.loc[sample_data['Submodel']==product, ['Submodel', 'Orders']]
    sample_data_p_q = df.resample('Q').agg({'Submodel':'first',
                                             'Orders': lambda x: np.nan if np.isnan(x).all() else np.sum(x)}).fillna(method='ffill')
    sample_data_q.append(sample_data_p_q)
    sample_data_p_m =df.sort_index().asfreq('MS').apply(lambda x: x.fillna(method='ffill') if x.name in ['Submodel'] else x.fillna(0))
    sample_data_m.append(sample_data_p_m)

sample_data_quarterly = pd.concat(sample_data_q)
sample_data_monthly = pd.concat(sample_data_m)
sample_data_quarterly.to_excel("Quarterly sample data.xlsx")
sample_data_monthly.to_excel("Monthly sample data.xlsx")

#Q1.b
# already filled na in subquestion a but still here is the function to fillna
def replace_na(data, freq='Month'):
    if data.isnull().values.any()==False:
        print("There are no NaN values in this data set")
    else:
        data_reset_index = data.reset_index().copy()
        if freq=='Month':
            data_reset_index['Orders'] = data_reset_index.groupby('Submodel').transform(lambda x: x.fillna(0))
        if freq=='Quarter':
            for product in products:
                if np.isnan(data.iloc[0,data.columns.get_loc('Orders')]):
                    data.iloc[0,data.columns.get_loc('Orders')]=0
                data.loc[data['Submodel']==product, 'Orders']=data.loc[data['Submodel']==product, 'Orders'].fillna(method='ffill')
    return data
sample_data_quarterly = replace_na(sample_data_quarterly, freq='Quarter')
sample_data_monthly = replace_na(sample_data_monthly, 'Monthly')

#Q1.c
# Discard the data beyond August 2022.
def discard_dates(data, date):
    data = data.loc[data.index<date,:]
    return data
sample_data_quarterly = discard_dates(sample_data_quarterly,'2022-08-01')
sample_data_monthly = discard_dates(sample_data_monthly,'2022-08-01')

#Calculate 6 months rolling average for trend, also fit regression line 
#ACF to detect seasonality
def get_stl_decomp(data):
    res = STL(data).fit()
    return res

def rolmean(data, window):
    rolmean = data.rolling(window).mean().dropna()
    return pd.Series(index=data.index, data=rolmean)
    
def get_trend(data):
    """
    Get the linear trend on the data which makes the time
    series non-stationary
    """
    n = len(data.index)
    X = np.reshape(np.arange(0, n), (n, 1))
    y = np.array(data)
    model = LinearRegression()
    model.fit(X, y)
    trend = model.predict(X)
    return pd.Series(index=data.index, data=trend)

def remove_trend(index, values):
    res = pd.Series(index=index, data=values)
    trend = get_trend(res)
    res = res.subtract(trend)
    return res, trend

def invert_trend(index, values, trend):
    res = pd.Series(index=index, data=values)
    assert len(res.index) == len(trend.index)                
    res = res + trend
    return res

def ts_analysis_plots(data, submodel, path="D:/Fujitsu onboargin project/Quarterly_Analysis", n_lags=6, window=6):

    def plot_cf(ax, fn, data, n_lags):
        """
        Plot autocorrelation functions for the loads
        """
        if fn==plot_acf:
            fn(data, ax=ax, lags=n_lags, color="#0504aa")
        else:
            fn(data, ax=ax, lags=n_lags, color="#0504aa",method='ywm')
        # for i in range(1, 5):
        #     ax.axvline(x=24*i, ymin=0.0, ymax=1.0, color='grey', ls="--")    

    # AD Fuller test and linear trend of the time series
    trend = get_trend(data)
    rol_mean = rolmean(data, window=window)
    stl_res_trend = get_stl_decomp(data).trend
    stl_res_season = get_stl_decomp(data).seasonal
    adf = adfuller(data)
            
    fig, axs = plt.subplots(2, 2, figsize=(25, 12))
    axs = axs.flat
    
    # original time series
    axs[0].plot(data, color='#0504aa', label='time series data')
    axs[0].plot(trend, color="red", label='regression line')
    axs[0].plot(rol_mean, color='green', label='rolling average')
    axs[0].plot(stl_res_trend, color='yellow', label='STL decomposition trend')
    axs[0].set(xlabel="Date", ylabel="Nr of Orders", 
               title=f"Number of Orders for {submodel} (ADF p-value: {round(adf[1], 6)})")
    axs[0].legend()
    # histogram of value distribution
    axs[1].plot(stl_res_season, color='#0504aa')
    axs[1].set(xlabel="Date", ylabel="SEASONAL", title="Seasonal Component")
    # autocorrelation function
    plot_cf(axs[2], plot_acf, stl_res_season, n_lags)
    axs[2].set(xlabel="lag", ylabel="ACF value")
    plt.tight_layout()
    # partial autocorrelation function
    plot_cf(axs[3], plot_pacf, stl_res_season, n_lags)
    axs[3].set(xlabel="lag", ylabel="PACF value")
    
    fig.savefig('{}/{}_analysis.png'.format(path, product))


for product in products:
    print(product)
    n_lags_q=math.floor((sample_data_quarterly.groupby('Submodel').count().loc[product, 'Orders']/2)-1)
    n_lags_m=math.floor((sample_data_monthly.groupby('Submodel').count().loc[product, 'Orders']/2)-1)
    ts_analysis_plots(sample_data_quarterly.loc[sample_data_quarterly['Submodel']==product,'Orders'], 
                        submodel=product,
                        path="D:/Fujitsu onboargin project/Quarterly_Analysis",
                        n_lags=n_lags_q,
                        window=3)
    ts_analysis_plots(sample_data_monthly.loc[sample_data_monthly['Submodel']==product,'Orders'],
                         submodel=product,
                         path="D:/Fujitsu onboargin project/Monthly_Analysis",
                         n_lags=n_lags_m,
                         window=6)
plt.close('all')

#Q1.d
def test_stationarity(data):
     #Perform Dickey-Fuller test:
    for product in products:
        data_p = data.loc[data['Submodel']==product, 'Orders']
        dftest = adfuller(data_p, autolag='AIC')
        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
        for key,value in dftest[4].items():
            dfoutput['Critical Value (%s)'%key] = value
        return dfoutput


# MAPE computation
def mape(y, yhat, perc=True):
    n = len(yhat.index) if type(yhat) == pd.Series else len(yhat)    
    mape = []
    for a, f in zip(y, yhat):
        # avoid division by 0
        if f > 1e-9:
            mape.append(np.abs((a - f)/a))
    mape = np.mean(np.array(mape))
    return mape * 100. if perc else mape

mape_scorer = make_scorer(mape, greater_is_better=False)
sample_data_quarterly_12 = sample_data_quarterly[sample_data_quarterly.groupby('Submodel')['Orders'].transform('count')>11]

def train_xgb(params, X_train, y_train):
    """
    Train XGBoost regressor using the parameters given as input. The model
    is validated using standard cross validation technique adapted for time series
    data. This function returns a friendly output for the hyperopt parameter optimization
    module.

    Parameters
    ----------
    params: dict with the parameters of the XGBoost regressor. For complete list see: 
            https://xgboost.readthedocs.io/en/latest/parameter.html
    X_train: pd.DataFrame with the training set features
    y_train: pd.Series with the training set targets    

    Returns
    -------
    dict with keys 'model' for the trained model, 'status' containing the hyperopt
    status string and 'loss' with the RMSE obtained from cross-validation
    """

    n_estimators = int(params["n_estimators"])
    max_depth= int(params["max_depth"])
    try:
        model = xgb.XGBRegressor(n_estimators=n_estimators, 
                                    max_depth=max_depth, 
                                    learning_rate=params["learning_rate"],
                                    subsample=params["subsample"])

        result = model.fit(X_train, 
                            y_train.values.ravel(),
                            eval_set=[(X_train, y_train.values.ravel())],
                            early_stopping_rounds=50)

        # cross validate using the right iterator for time series
        cv_space = TimeSeriesSplit(n_splits=5)
        cv_score = cross_val_score(model, 
                                    X_train, y_train.values.ravel(),
                                    cv=cv_space, 
                                    scoring=mape_scorer,
                                    error_score='raise')
        rmse = np.abs(np.mean(np.array(cv_score)))
        return {
            "loss": rmse,
            "status": STATUS_OK,
            "model": model
        }
    except ValueError as ex:
        return {
            "error": ex,
            "status": STATUS_FAIL
        }

def optimize_xgb(X_train, y_train, max_evals=10):
    """
    Run Bayesan optimization to find the optimal XGBoost algorithm
    hyperparameters.
    
    Parameters
    ----------
    X_train: pd.DataFrame with the training set features
    y_train: pd.Series with the training set targets
    max_evals: the maximum number of iterations in the Bayesian optimization method
    
    Returns
    -------
    best: dict with the best parameters obtained
    trials: a list of hyperopt Trials objects with the history of the optimization
    """
    
    space = {
        "n_estimators": hp.quniform("n_estimators", 100, 1000, 10),
        "max_depth": hp.quniform("max_depth", 1, 8, 1),
        "learning_rate": hp.loguniform("learning_rate", -5, 1),
        "subsample": hp.uniform("subsample", 0.8, 1),
        "gamma": hp.quniform("gamma", 0, 100, 1)
    }

    objective_fn = partial(train_xgb, 
                           X_train=X_train, 
                           y_train=y_train)
    
    trials = Trials()
    best = fmin(fn=objective_fn,
                space=space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials)

    # evaluate the best model on the test set
    print(f"""
    Best parameters:
        learning_rate: {best["learning_rate"]} 
        n_estimators: {best["n_estimators"]}
        max_depth: {best["max_depth"]}
        sub_sample: {best["subsample"]}
        gamma: {best["gamma"]}
    """)
    return best, trials

def create_lag_features(y, n_lags):
    
    scaler = StandardScaler()
    features = pd.DataFrame()
    df = pd.DataFrame()
    
    for l in range(1, n_lags):
        df[f"lag_{l}"] = y.shift(l)
    features = pd.DataFrame(scaler.fit_transform(df[df.columns]),
                            columns=df.columns)
    # features = pd.DataFrame(df[df.columns],
    #                         columns=df.columns)
    features.index = y.index
    return features

def create_ts_features(data):
    
    features = pd.DataFrame()
    
    features["season"] = (data.index.month%12 + 3)//3
    features.index = data.index
    features["season"] = features["season"].astype("category")
    return features

def encode_categorical(data):
    ohe = OneHotEncoder(drop='first')

    data_train_object = data.select_dtypes('category')
    ohe.fit(data_train_object)
    codes = ohe.transform(data_train_object).toarray()
    feature_names = ohe.get_feature_names_out()
    data_ = pd.concat([data.select_dtypes(exclude='category'), 
                        pd.DataFrame(codes, columns=feature_names, index=data.index).astype(int)], axis=1)
    return data_, ohe
 
"""
for product in products[0:1]:
    ts_features = create_ts_features(sample_data_quarterly_12.loc[sample_data_quarterly_12['Submodel']==product, 'Orders'])
    lag_features = create_lag_features(sample_data_quarterly_12.loc[sample_data_quarterly_12['Submodel']==product,'Orders'], 2)
    features = ts_features.join(lag_features, how='outer').dropna()
    
    X_train = features.iloc[:-4,:]
    X_train, ohe = encode_categorical(X_train)
    y_train = sample_data_quarterly_12.loc[sample_data_quarterly_12['Submodel']==product,'Orders'][1:-4]
    # y_train, y_train_trend = remove_trend(y_train.index, y_train.values)

    y_test = sample_data_quarterly_12.loc[sample_data_quarterly_12['Submodel']==product,'Orders'][-4:]
    xgb_model = {}
    best, trials = optimize_xgb(X_train, y_train, max_evals=5)
    res = train_xgb(best, X_train, y_train)
    xgb_model[product] = res["model"]
    
    # lags used in building the features for the one-step ahead model
    feature_lags = [int(f.split("_")[1]) for f in features if "lag" in f]
"""
def recursive_forecast(y, cols, ohe, model, lags, 
                       n_steps=13, step="1Q"):
    
    """
    Parameters
    ----------
    y: pd.Series holding the input time-series to forecast
    model: pre-trained machine learning model
    lags: list of lags used for training the model
    n_steps: number of time periods in the forecasting horizon
    step: forecasting time period
    
    Returns
    -------
    fcast_values: pd.Series with forecasted values 
    """
    
    # get the dates to forecast
    last_date = y.index[-1] + pd.DateOffset(months=3)
    fcast_range = pd.date_range(last_date,
                                periods=n_steps, 
                                freq=step)
    fcasted_values = []

    X_test_ = pd.DataFrame(index=fcast_range, columns=cols)  
    target = y.copy()
    for date in fcast_range:
        ts_features = create_ts_features(X_test_)        
        new_point = fcasted_values[-1] if len(fcasted_values) > 0 else 0.0   
        target = target.append(pd.Series(index=[date], data=new_point))
        if len(lags) > 0:
            lags_features = create_lag_features(target, n_lags=lags[0]+1)
            features = pd.concat([ts_features, lags_features], axis=1, join="inner").dropna()
        else:
            features = ts_features
        codes = ohe.transform(features.select_dtypes('category')).toarray()
        feature_names = ohe.get_feature_names_out()
        X_test = pd.concat([features.select_dtypes(exclude='category'), 
                            pd.DataFrame(codes, columns=feature_names, index=features.index).astype(int)], axis=1)
        # forecast
        predictions = model.predict(X_test)
        fcasted_values.append(predictions[-1])
        target[date]=predictions[-1]
        
    return pd.Series(index=fcast_range, data=fcasted_values)

"""
forecasts = recursive_forecast(y=y_train, cols=X_train.columns, ohe=ohe, model=xgb_model[products[0]], lags=feature_lags)
print("True values \n",sample_data_quarterly_12.loc[sample_data_quarterly_12['Submodel']==products[0],'Orders'][-4:], " \n forecasts \n", forecasts)
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.plot(sample_data_quarterly_12.loc[sample_data_quarterly_12['Submodel']==products[0],'Orders'], label="target", color="blue")
ax.plot(forecasts, label="predicted", color="red")
ax.legend()
plt.show()
"""
# Q1.d
from pmdarima.arima import auto_arima
import pmdarima as pm
from bokeh.plotting import figure, show, output_notebook

def plot_arima(truth, forecasts):
    
    # forecasts = pd.Series(forecasts, index=data_.index)
    
    # set up the plot
    ax= plt.axes()
    ax.plot(truth, color='yellow', label='True values')
    ax.plot(forecasts,color='red', label='Forecast values')
    ax.set(xlabel="Date", ylabel="Nr of Orders", 
               title=f"Number of Orders")
    ax.legend()
    
    return

for product in products:
    data_ = sample_data_monthly.loc[sample_data_monthly['Submodel']==product,'Orders']
    train = data_[:-6]
    test = data_[-6:]
    stepwise_fit = auto_arima(train,
                            test='adf',
                            start_p=0, d=0, start_q=1,
                            max_p=3, max_d=3, max_q=3,
                            start_P=0, D=0, start_Q=0,
                            max_P=3, max_D=3, max_Q=3,
                            m=12,
                            seasonal=False,
                            trace=True,
                            error_action='ignore',  # don't want to know if an order does not work
                            suppress_warnings=True,  # don't want convergence warnings
                            stepwise=False)
    print(stepwise_fit.summary())
    in_sample_preds = stepwise_fit.predict_in_sample()
    forecasts = pd.Series(stepwise_fit.predict(n_periods=6), index=test.index)
    plot_arima(data_, forecasts)
    plt.show()
    # arima = {}
    # arima[product] = 