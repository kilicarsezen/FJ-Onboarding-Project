import matplotlib.pyplot as plt
from pmdarima.arima import auto_arima
import pmdarima as pm

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

def get_monthly_forcasts(data, m, seasonal):    
    if seasonal:
        stepwise_fit = auto_arima(data,
                                test='adf',
                                m=m,
                                D=1,
                                d=1,
                                max_order=5,
                                seasonal=True,
                                trace=True,
                                error_action='ignore',  # don't want to know if an order does not work
                                suppress_warnings=True,  # don't want convergence warnings
                                stepwise=False)
    else:
        stepwise_fit = auto_arima(data,
                                test='adf',
                                max_order=5,
                                seasonal=False,
                                trace=True,
                                error_action='ignore',  # don't want to know if an order does not work
                                suppress_warnings=True,  # don't want convergence warnings
                                stepwise=False)

    print(stepwise_fit.summary())
    in_sample_preds = stepwise_fit.predict_in_sample()
    forecasts = stepwise_fit.predict(n_periods=9).values    
    return forecasts