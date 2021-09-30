# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 14:36:21 2021

@author: ADMIN
"""

### AUTO SARIMAX MODEL ###

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error

states =['Malaysia']
 
df_state_recs =[]
stateName = states[0]
df_per_State_features = pd.read_csv('Data/' + stateName +'.csv')
df_per_State_features = df_per_State_features.fillna(0)
df_per_State_features["Active Cases"].replace({0:1}, inplace=True)
df_state_recs.append(df_per_State_features)
    

# Train Set
model_scores=[]
df_per_State_features = df_state_recs[0]
stateName = states[0]

data = df_per_State_features['Active Cases'].astype('double').values
daterange = df_per_State_features['Date'].values
no_Dates = len(daterange)

dateStart = daterange[0]
dateEnd = daterange[no_Dates - 1]

date_index= pd.date_range(start=dateStart, end=dateEnd, freq='D')

activecases = pd.Series(data, date_index)
df_per_State_sel_features = df_per_State_features.copy(deep=False)

df_per_State_sel_features["Days Since"]=date_index-date_index[0]
df_per_State_sel_features["Days Since"]=df_per_State_sel_features["Days Since"].dt.days


df_per_State_sel_features = df_per_State_sel_features.iloc[:,[4,5, 7,8,9,10,11,12,13,14,15,16,23]]
df_per_State_sel_features.head()


df_per_State_features["Days Since"]=date_index-date_index[0]
df_per_State_features["Days Since"]=df_per_State_features["Days Since"].dt.days

train_ml=df_per_State_features.iloc[:int(df_per_State_features.shape[0]*0.780)]
valid_ml=df_per_State_features.iloc[int(df_per_State_features.shape[0]*0.780):]

totActiveCases = activecases.values.reshape(-1,1)
trainActiveCases =totActiveCases[0:int(df_per_State_features.shape[0]*0.780)]
validActiveCases=totActiveCases[int(df_per_State_features.shape[0]*0.780):]

train_dates = df_per_State_features['Date'].iloc[:int(df_per_State_sel_features.shape[0]*0.780)].values
valid_dates = df_per_State_features['Date'].iloc[int(df_per_State_sel_features.shape[0]*0.780):].values

# Modeling
model_arima= auto_arima(trainActiveCases,start_p=1, start_q=1,
                         test='adf',
                         max_p=3, max_q=3, m=12,
                         start_P=0, seasonal=True,
                         d=None, D=1, trace=True,
                         error_action='ignore',  
                         suppress_warnings=True, 
                         stepwise=True)
model_arima_fit = model_arima.fit(trainActiveCases)
prediction_arima=model_arima_fit.predict(len(validActiveCases))

print(model_arima_fit.summary())
plt.figure(figsize=(10,10))

# plot residual errors
residuals = pd.DataFrame(model_arima_fit.resid())
model_arima_fit.plot_diagnostics()
plt.show()
residuals.plot(kind='kde')
plt.show()

# Root Mean Square Error
model_scores.append(np.sqrt(mean_squared_error(validActiveCases,prediction_arima)))
print("Root Mean Square Error for Auto ARIMA Model: ",np.sqrt(mean_squared_error(validActiveCases,prediction_arima)))

# Active Cases Forecasting
index= pd.date_range(start=train_dates[0], periods=len(train_dates), freq='D')
valid_index = pd.date_range(start=valid_dates[0], periods=len(valid_dates), freq='D')

train_active =  pd.Series(train_ml['Active Cases'].values, index)
valid_active =  pd.Series(valid_ml['Active Cases'].values, valid_index)
pred_active =   pd.Series(prediction_arima, valid_index)

f, ax = plt.subplots(1,1 , figsize=(12,10))
plt.plot(train_active, marker='o',color='blue',label ="Train Data Set")
plt.plot(valid_active, marker='o',color='green',label ="Valid Data Set")
plt.plot(pred_active, marker='o',color='red',label ="Predicted Auto ARIMA")

plt.legend()
plt.xlabel("Date Time")
plt.ylabel('Active Cases')
plt.title("Active Cases Auto SARIMAX Model Forecasting for Malaysia")

# Forecast 
n_periods = 47
fc, confint = model_arima_fit.predict(n_periods=n_periods, return_conf_int=True)
valid_index = pd.date_range(start=valid_dates[0], periods=n_periods, freq='D')


train_active =  pd.Series(train_ml['Active Cases'].values, index)

# make series for plotting purpose
fc_series = pd.Series(fc, index=valid_index)
print('Forecast Value For the Subsequent 30 Days :')
print(fc_series)

lower_series = pd.Series(confint[:, 0], index=valid_index)
upper_series = pd.Series(confint[:, 1], index=valid_index)

# Plot
f, ax = plt.subplots(1,1 , figsize=(12,10))
plt.plot(train_active, color='blue')
plt.plot(fc_series, color='darkgreen')
plt.fill_between(lower_series.index, 
                 lower_series, 
                 upper_series, 
                 color='k', alpha=.15)

plt.title("Daily New Cases of Covid-19 in Malaysia")
plt.xlabel('Date')
plt.ylabel('Population')
plt.savefig('Covid-19 New Cases Forecast.png')
plt.show()

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

cred = credentials.Certificate(r"C:\Users\ADMIN\Desktop\Python\Auto Arima\draft-7bfc6-7bc4dd407ac1.json")
firebase_admin.initialize_app(cred)
user = firestore.client()

x=fc_series.to_dict()
keys_values = x.items()

new_x = {str(key): str(value) for key, value in keys_values}

def save(collection_id,document_id,Data):
    user.collection(collection_id).document(document_id).set(Data)
    
save(
    collection_id = "Predicted Daily New Cases",
    document_id = "Daily New Cases Forecast",
    Data = new_x
)