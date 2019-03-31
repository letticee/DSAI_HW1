#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 03:27:39 2019

@author: waiting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dateparse = lambda x: pd.datetime.strptime(x, '%Y%m%d')
with open('170101_190131過去電力供需資訊.csv', 'r') as f:
    data = pd.read_csv(f, parse_dates=['日期'], date_parser=dateparse, index_col=0)
data['Weekday'] = data.index.weekday
y_data = pd.Series(data['尖峰負載(MW)'],index=data.index)

plt.figure(figsize = (20,5))
plt.plot(y_data['2017'])
plt.title('2017 Peaking Power')
plt.figure(figsize = (20,5))
plt.plot(y_data['2018'])
plt.title('2018 Peaking Power')

#%%
dateparse = lambda x: pd.datetime.strptime(x, '%Y/%m/%d')
with open('Taipei_temperature.csv', 'r') as f:   
     weather = pd.read_csv(f, parse_dates=['date'], date_parser=dateparse, index_col=0)
temper = pd.Series(weather['Temperature'],index=weather.index)
max_temper = pd.Series(weather['T Max'],index=weather.index)

fig17 = plt.figure(figsize = (20,5))
ax1 = fig17.add_subplot(111)
ax1.plot(y_data['2017'],color='steelblue',label='Peaking Power')
ax1.legend(loc='upper right')
ax2 = ax1.twinx()
ax2.plot(temper['2017'],color='coral',label='Temperature')
ax2.legend(loc='upper left')
plt.title('2017 Peaking Power and Temperature')

fig18 = plt.figure(figsize = (20,5))
ax1 = fig18.add_subplot(111)
ax1.plot(y_data['2018'],color='steelblue',label='Peaking Power')
ax1.legend(loc='upper right')
ax2 = ax1.twinx()
ax2.plot(temper['2018'],color='coral',label='Temperature')
ax2.legend(loc='upper left')
plt.title('2018 Peaking Power and Temperature')

temper['2017'].corr(y_data['2017'])
temper['2018'].corr(y_data['2018'])
max_temper['2017'].corr(y_data['2017'])
max_temper['2018'].corr(y_data['2018'])

#%%
name = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
y_week = [None] * 7
plt.figure(figsize = (20,5))
y_data1718 = y_data['2017':'2018']
temper1718 = temper['2017':'2018']
for day in range(7):
    y_week[day] = y_data1718[data['Weekday'] == day]  
    plt.plot(y_week[day], label=name[day])
plt.legend(loc='upper right')   
plt.title('Peaking Power on Different Weekday')

temp_week = [None] * 7

for day in range(7):
    y_week[day] = y_data1718[data['Weekday'] == day]  
    temp_week[day] = temper1718[data['Weekday'] == day]  
    fig = plt.figure(figsize = (20,5))
    ax1 = fig.add_subplot(111)
    ax1.plot(y_week[day],color='steelblue',label=name[day])
    ax1.legend(loc='upper right')
    ax2 = ax1.twinx()
    ax2.plot(temp_week[day],color='coral',label=name[day]+'_temp')
    ax2.legend(loc='upper left')
    plt.title('Peaking Power and Temperature on Different Weekday on '+name[day])
    
    print(name[day]," ",y_week[day].corr(temp_week[day]))

    
#%% linear regression
from sklearn.linear_model import LinearRegression

lm_trainX = [None] * 7
lm_trainY = [None] * 7
lm = [None] * 7
plt.figure(figsize = (20,20))
for day in range(7):
    lm_trainX[day] = np.reshape(temp_week[day].values,(len(temp_week[day].values), 1))
    lm_trainY[day] = np.reshape(y_week[day].values,(len(y_week[day].values), 1))
    lm[day] =  LinearRegression()
    lm[day].fit(lm_trainX[day], lm_trainY[day])
        
    plt.subplot(4, 3, day+1)
    plt.scatter(lm_trainX[day], lm_trainY[day])
    plt.plot( lm_trainX[day], lm[day].predict( lm_trainX[day]), color='black')
    
    plt.title('Peaking Power and Temperature on '+name[day])

#%%
from sklearn.preprocessing import PolynomialFeatures

lm_trainX = [None] * 7
lm_trainX2 = [None] * 7
lm_trainY = [None] * 7
lm2 = [None] * 7
poly = PolynomialFeatures(degree=2)
plt.figure(figsize = (20,20))
for day in range(7):
    lm_trainX[day] = np.reshape(temp_week[day].values,(len(temp_week[day].values), 1))
    lm_trainY[day] = np.reshape(y_week[day].values,(len(y_week[day].values), 1))
    poly.fit(lm_trainX[day])
    lm_trainX2[day] = poly.transform(lm_trainX[day])
    lm2[day] =  LinearRegression()
    lm2[day].fit(lm_trainX2[day], lm_trainY[day])
    
    y_predict2 = lm2[day].predict(lm_trainX2[day])
    plt.subplot(4, 3, day+1)
    plt.scatter(lm_trainX[day], lm_trainY[day])
    plt.plot( np.sort(temp_week[day].values), y_predict2[np.argsort(temp_week[day].values)], color='black')
    
    plt.title('Peaking Power and Temperature on '+name[day])   
#%% 
y_workday =  y_data1718[(data['Weekday'] >= 0) & (data['Weekday'] <= 4)]
temp_workday = temper1718[(data['Weekday'] >= 0) & (data['Weekday'] <= 4)]  

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly = PolynomialFeatures(degree=2)

lm_workday_trainX = np.reshape(temp_workday.values,(len(temp_workday.values), 1))
lm_workday_trainY = np.reshape(y_workday.values,(len(y_workday.values), 1))
poly.fit(lm_workday_trainX)
lm_workday_trainX2 = poly.transform(lm_workday_trainX)
lm_workday2 =  LinearRegression()
lm_workday2.fit(lm_workday_trainX2, lm_workday_trainY)
y_predict = lm_workday2.predict(lm_workday_trainX2)
plt.figure()
plt.scatter(lm_workday_trainX, lm_workday_trainY)
plt.plot( np.sort(temp_workday.values), y_predict[np.argsort(temp_workday.values)], color='black')
 
plt.title('Polynomial Regression of Peaking Power and Temperature on Workday') 

#%%
y_weekend =  y_data1718[data['Weekday'] >=5]
temp_weekend = temper1718[data['Weekday'] >=5]  

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly = PolynomialFeatures(degree=2)

lm_weekend_trainX = np.reshape(temp_weekend.values,(len(temp_weekend.values), 1))
lm_weekend_trainY = np.reshape(y_weekend.values,(len(y_weekend.values), 1))
poly.fit(lm_weekend_trainX)
lm_weekend_trainX2 = poly.transform(lm_weekend_trainX)
lm_weekend2 =  LinearRegression()
lm_weekend2.fit(lm_weekend_trainX2, lm_weekend_trainY)
y_predict = lm_weekend2.predict(lm_weekend_trainX2)
plt.figure()
plt.scatter(lm_weekend_trainX, lm_weekend_trainY)
plt.plot( np.sort(temp_weekend.values), y_predict[np.argsort(temp_weekend.values)], color='black')
 
plt.title('Polynomial Regression of Peaking Power and Temperature on weekend') 
    
 #%% 
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

y_data19 = y_data['2019']
temper19 = temper['2019']
y_week19 = [None] * 7
temp_week19 = [None] * 7

lm_testX = [None] * 7
lm_testX2 = [None] * 7
lm_testY = [None] * 7
y_pred_test = [None] * 7
err_poly = [None] * 7
for day in range(7):
    y_week19[day] = y_data19[data['Weekday'] == day]  
    temp_week19[day] = temper19[data['Weekday'] == day] 
    
    lm_testX[day] = np.reshape(temp_week19[day].values,(len(temp_week19[day].values), 1))
    lm_testY[day] = np.reshape(y_week19[day].values,(len(y_week19[day].values), 1))
    poly.fit(lm_testX[day])
    lm_testX2[day] = poly.transform(lm_testX[day])
    y_pred_test[day] = lm2[day].predict(lm_testX2[day])
    
    err_poly[day] = rmse(y_pred_test[day].T,y_week19[day].values.T)
        
#%% 
err_lr = [None] * 7 
for day in range(7):
    lm_testX[day] = np.reshape(temp_week19[day].values,(len(temp_week19[day].values), 1))
    lm_testY[day] = np.reshape(y_week19[day].values,(len(y_week19[day].values), 1))
    y_pred_test[day] = lm[day].predict(lm_testX[day])
    
    err_lr[day] = rmse(y_pred_test[day].T,y_week19[day].values.T)


#%% 
y_workday19 = y_data19[(data['Weekday'] >= 0) & (data['Weekday'] <= 4)]  
temp_workday19 = temper19[(data['Weekday'] >= 0) & (data['Weekday'] <= 4)] 
    
lm_workday_testX = np.reshape(temp_workday19.values,(len(temp_workday19.values), 1))
lm_workday_testY = np.reshape(y_workday19.values,(len(y_workday19.values), 1))
poly.fit(lm_workday_testX)
lm_workday_testX2 = poly.transform(lm_workday_testX)
y_pred_test = lm_workday2.predict(lm_workday_testX2)
    
err_workday = rmse(y_pred_test.T,y_workday19.values.T)

#%%
y_weekend19 = y_data19[data['Weekday'] >= 5]  
temp_weekend19 = temper19[data['Weekday'] >= 5] 
    
lm_weekend_testX = np.reshape(temp_weekend19.values,(len(temp_weekend19.values), 1))
lm_weekend_testY = np.reshape(y_weekend19.values,(len(y_weekend19.values), 1))
poly.fit(lm_weekend_testX)
lm_weekend_testX2 = poly.transform(lm_weekend_testX)
y_pred_test = lm_weekend2.predict(lm_weekend_testX2)
    
err_weekend = rmse(y_pred_test.T,y_weekend19.values.T)

#%%
dateparse = lambda x: pd.datetime.strptime(x, '%Y%m%d')
with open('201904_temp.csv', 'r') as f:   
     weather1904 = pd.read_csv(f, parse_dates=['date'], date_parser=dateparse, index_col=0)
max_temper1904 = pd.Series(weather1904['T Max'],index=weather1904.index)
min_temper1904 = pd.Series(weather1904['T Min'],index=weather1904.index)
temper1904 = (max_temper1904+min_temper1904)/2
weekdayr1904 = weather1904.index.weekday
weather1904.head(7)   

weekdayr1904.values[2] = 5
weekdayr1904.values[3] = 6

x_1904 = [None] * len(weather1904)
temper1904_2 = [None] * len(weather1904)
y_pred_1904 = np.zeros(len(weather1904))
for day in range(len(weather1904)):
    x_1904[day] = np.reshape(temper1904[day], (1, -1))
    poly.fit(x_1904[day])
    temper1904_2[day] = poly.transform(x_1904[day])
    y_pred_1904[day] = lm2[weekdayr1904[day]].predict(temper1904_2[day])
    
err_pred = [None] * 7 
yy = [28700,28600,25700,24600,24300,24500,28500]
for day in range(len(weather1904)):
    err_pred[day] = rmse(y_pred_1904[day].T,yy[day])

#%%
result = pd.DataFrame()
result['peak_load(MW)'] = (y_pred_1904).astype(np.int)
result.index = weather1904.index
 
with open('submission.csv', 'w', newline='\n') as f:
    result.to_csv(f,date_format='%Y%m%d')  

    




