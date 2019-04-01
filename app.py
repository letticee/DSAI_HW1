#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 03:27:39 2019

@author: waiting
"""

import numpy as np
import pandas as pd

dateparse = lambda x: pd.datetime.strptime(x, '%Y%m%d')
with open('170101_190131過去電力供需資訊.csv', 'r') as f:
    data = pd.read_csv(f, parse_dates=['日期'], date_parser=dateparse, index_col=0)
data['Weekday'] = data.index.weekday
y_data = pd.Series(data['尖峰負載(MW)'],index=data.index)

dateparse = lambda x: pd.datetime.strptime(x, '%Y/%m/%d')
with open('Taipei_temperature.csv', 'r') as f:   
     weather = pd.read_csv(f, parse_dates=['date'], date_parser=dateparse, index_col=0)
temper = pd.Series(weather['Temperature'],index=weather.index)
min_temper = pd.Series(weather['T Min'],index=weather.index)
max_temper = pd.Series(weather['T Max'],index=weather.index)

temp_week = [None] * 7
y_week = [None] * 7
y_data1718 = y_data['2017':'2018']
min_temper1718 = min_temper['2017':'2018']
for day in range(7):
    y_week[day] = y_data1718[data['Weekday'] == day]  
    temp_week[day] = min_temper1718[data['Weekday'] == day] 

#%% train every day model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

lm_trainX = [None] * 7
lm_trainX2 = [None] * 7
lm_trainY = [None] * 7
lm2 = [None] * 7
poly = PolynomialFeatures(degree=2)
for day in range(7):
    lm_trainX[day] = np.reshape(temp_week[day].values,(len(temp_week[day].values), 1))
    lm_trainY[day] = np.reshape(y_week[day].values,(len(y_week[day].values), 1))
    poly.fit(lm_trainX[day])
    lm_trainX2[day] = poly.transform(lm_trainX[day])
    lm2[day] =  LinearRegression()
    lm2[day].fit(lm_trainX2[day], lm_trainY[day])
    
    y_predict2 = lm2[day].predict(lm_trainX2[day])

#%% train weekday model
y_workday =  y_data1718[(data['Weekday'] >= 0) & (data['Weekday'] <= 4)]
temp_workday = min_temper1718[(data['Weekday'] >= 0) & (data['Weekday'] <= 4)]  

poly = PolynomialFeatures(degree=2)

lm_workday_trainX = np.reshape(temp_workday.values,(len(temp_workday.values), 1))
lm_workday_trainY = np.reshape(y_workday.values,(len(y_workday.values), 1))
poly.fit(lm_workday_trainX)
lm_workday_trainX2 = poly.transform(lm_workday_trainX)
lm_workday2 =  LinearRegression()
lm_workday2.fit(lm_workday_trainX2, lm_workday_trainY)
y_predict = lm_workday2.predict(lm_workday_trainX2)
    
 #%% predict
dateparse = lambda x: pd.datetime.strptime(x, '%Y%m%d')
with open('201904_temp_gov.csv', 'r') as f:   
     weather1904 = pd.read_csv(f, parse_dates=['date'], date_parser=dateparse, index_col=0)
max_temper1904 = pd.Series(weather1904['T Max'],index=weather1904.index)
min_temper1904 = pd.Series(weather1904['T Min'],index=weather1904.index)
weekdayr1904 = weather1904.index.weekday
weekdayr1904.values[2] = 5
weekdayr1904.values[3] = 6
weekdayr1904.values[4] = 6

x_1904 = [None] * len(weather1904)
min_temper1904_2 = [None] * len(weather1904)
y_pred_1904 = np.zeros(len(weather1904))
ori = []
pred = []
for day in range(len(weather1904)):
    x_1904[day] = np.reshape(min_temper1904[day], (1, -1))
    poly.fit(x_1904[day])
    min_temper1904_2[day] = poly.transform(x_1904[day])
    #if weekdayr1904[day] < 5:
        #y_pred_1904[day] = lm_workday2.predict(min_temper1904_2[day])
    #else:
    y_pred_1904[day] = lm2[weekdayr1904[day]].predict(min_temper1904_2[day])

result = pd.DataFrame()
result['peak_load(MW)'] = (y_pred_1904).astype(np.int)
result.index = weather1904.index

with open('submission.csv', 'w', newline='\n') as f:
    result.to_csv(f,date_format='%Y%m%d')