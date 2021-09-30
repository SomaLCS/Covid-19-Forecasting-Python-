# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 11:52:44 2021

@author: ADMIN
"""

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from pandas import datetime
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf




def parser(s):
    return datetime.strptime(s, '%Y-%m-%d')

#read data
covid_activecases = pd.read_csv('train.csv', parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

#infer the frequency of the data
covid_activecases = covid_activecases.asfreq(pd.infer_freq(covid_activecases.index))

#Setting the range of time
start_date = datetime(2019,12,31)
end_date = datetime(2021,4,9)
total_covid_activecases = covid_activecases[start_date:end_date]

#Plotting
plt.figure(figsize=(10,4))
plt.plot(total_covid_activecases)
plt.title('Active Cases in Malaysia', fontsize=20)
plt.ylabel('Active Cases', fontsize=16)

#Train & Test
total_covid_activecases_train = total_covid_activecases[:48]
total_covid_activecases_test = total_covid_activecases[47:60]
print("train = " + str(total_covid_activecases_train.size) +" month")
print("test = " + str(total_covid_activecases_test.size) +" month")

from pmdarima import auto_arima

model = auto_arima(total_covid_activecases_train, start_p=1, start_q=1, m=12, seasonal=False,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)
print(model.aic())


