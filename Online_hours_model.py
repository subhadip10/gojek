#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 11:21:13 2019

@author: s0p00zp
"""

import pandas as pd
import numpy as np
import seaborn as sns
import datetime
import time
import sys
import matplotlib.pyplot as plt

#Specifying the file details
path='/Users/s0p00zp/.kaggle/DS TakeHome Assignment/data'
pings='pings.csv'
driver='drivers.csv'
test='test.csv'

pings_df=pd.read_csv(path+'/'+pings)
drivers_df=pd.read_csv(path+'/'+driver)
test_df=pd.read_csv(path+'/'+test)


print(pings_df.head())

print("datatypes in pings: \n",pings_df.dtypes)

print("checking for missing values in data: \n",pings_df.isna().any())

print("Rows/columns: \n",pings_df.shape)

print("number of Driver's id in ping data:",len(set(pings_df.driver_id)))

#checking differnce between drivers in driver data and ping data
print("Number of driver Ids not present in pings:",len(set(drivers_df.driver_id)-set(pings_df.driver_id)))

print("converting unix epochs time stamp to datetime format and extracting date,time")

def Extracting_Date(pings_df):
    pings_df['date']=[time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime(int(x)+25200)) for x in pings_df.ping_timestamp]
    pings_df['date']=pd.to_datetime(pings_df['date'])
    pings_df['hour']=pings_df['date'].dt.hour
    pings_df['date_month']=pings_df['date'].dt.date
    return(pings_df)

pings_df=Extracting_Date(pings_df)


print("Checking for duplicate rows in training data: \n")
duplicate=pings_df.duplicated()
print("number of duplicate records:",np.sum(duplicate))

print("removing duplicate records")
pings_df_clean=pings_df[~duplicate]
print("check difference in shape for clean and unclean data: \n")
print(pings_df_clean.shape[0],pings_df.shape[0])

def calculate_online_hours(pings_df):
    
    #creating features from date column
    pings_df['hour']=pings_df['date'].dt.hour
    pings_df['date_month']=pings_df['date'].dt.date
    
    #selecting columns which are relevant
    pings_df_s=pings_df[['driver_id','date','date_month','hour']]
    
    #1st grouping with respect to driver_id ,date and hour of that date.
    pings_df_time=pings_df_s.groupby(['driver_id','date_month','hour'])['date'].count()
    #Calculating number of pings in a specific hours of aday
    pings_df_time=pings_df_time.reset_index()
    #pings_df_time['date']=pings_df_time['date']-1
    
    #Multiplying number of pings with 15 since each ping is at an interval of 15 sec
    #dividing the sum of all seconds by 3600    
    pings_df_hrs=pings_df_time.groupby(['driver_id','date_month'])['date'].apply(lambda x:(x.sum()*15)/3600)
    pings_df_hrs=pings_df_hrs.reset_index()
    
    #Rounding of the hours to calculate online_hours
    #pings_df_hrs['online_hours']=pings_df_hrs['date'].round()
    pings_df_hrs['online_hours']=np.floor(pings_df_hrs['date'])
    #dropping the date column from the data frame
    pings_df_hrs=pings_df_hrs.drop(['date'],axis=1)
    
    #creating features for date
    pings_df_hrs['dayofyear']=pings_df_hrs['date_month'].dt.dayofyear
    pings_df_hrs['dayofweek']=pings_df_hrs['date_month'].dt.dayofweek
    pings_df_hrs['dayofmonth']=pings_df_hrs['date_month'].dt.day
    
    return(pings_df_hrs)

train_data=calculate_online_hours(pings_df_clean)

tmp=calculate_online_hours(pings_df)
plt.figure()
sns.boxplot(tmp.online_hours).set_title("boxplot for unclean hours")

plt.figure()
sns.boxplot(train_data.online_hours).set_title("boxplot for clean hours")

#creating a baseline model by predicting online_hours as mean
def max_min(x):
    return(x.max()-x.min())
def percentile_1(x):
    return(np.percentile(x,0.25))
def percentile_2(x):
    return(np.percentile(x,0.75))
def percentile_3(x):
    return(np.percentile(x,0.50))
def percentile_4(x):
    return(np.percentile(x,0.95))
def percentile_5(x):
    return(np.percentile(x,0.10))
f={'online_hours':['mean','min','max',
                   max_min,percentile_1,percentile_2,
                   percentile_3,percentile_4,percentile_5]}

#train_df_mean=train_data.groupby(['driver_id'])['online_hours'].mean()
train_df_mean=train_data.groupby(['driver_id']).agg(f)
train_df_mean=train_df_mean.reset_index()
train_df_mean.columns=train_df_mean.columns.map('_'.join)
train_df_mean=train_df_mean.rename(columns={"driver_id_":"driver_id"})
#train_df_mean=train_df_mean.drop(['index'],axis=1)
#train_df_mean['online_hours_mean']=train_df_mean['online_hours_mean'].round()
all_data_M=pd.merge(train_data.drop_duplicates(),drivers_df.drop_duplicates(),how='inner',on='driver_id')
sns.boxplot(all_data_M.gender,all_data_M.online_hours).set_title("online_hours vs gender")
test_df['date']=pd.to_datetime(test_df['date'])
def Creating_train_test(train_data,test_df,drivers_df,label_encode=0,merge=0):
    #creating features for date
    test_df['dayofyear']=test_df['date'].dt.dayofyear
    test_df['dayofweek']=test_df['date'].dt.dayofweek
    test_df['dayofmonth']=test_df['date'].dt.day
    train_data=train_data.rename(columns={'date_month':'date'})
    train_data['ind']='tr'
    test_df['ind']='te'
    all_data=pd.concat([train_data,test_df],axis=0)
    #merging training data with driver data
    all_data_M=pd.merge(all_data.drop_duplicates(),drivers_df.drop_duplicates(),how='inner',on='driver_id')
    gender=all_data_M.groupby(['gender'])['online_hours'].mean().reset_index()
    gender=gender.rename(columns={'online_hours':'mean_gender_hours'})
    all_data_M=pd.merge(all_data_M,gender,how='left',on='gender')
    if merge==0:
        all_data_M=pd.merge(all_data_M,train_df_mean,how="left",on="driver_id")
        all_data_M=all_data_M.fillna(0)
    if label_encode==0:
        x=pd.get_dummies(all_data_M,columns=['driver_id','gender','dayofweek'])
    else:
        x=pd.get_dummies(all_data_M,columns=['gender','dayofweek'])
        #x['driver_id']=LabelEncoder().fit_transform(all_data_M.driver_id)
    x=x.drop(['date'],axis=1)
    x_train=x[x.ind=='tr']
    x_test=x[x.ind=='te']
    x_train=x_train.drop(['ind'],axis=1)
    x_test=x_test.drop(['ind'],axis=1)
    return(x_train,x_test)
    
x_train,x_test=Creating_train_test(train_data,test_df,drivers_df,label_encode=1,merge=0)
#x_test.online_hours_mean=x_test.online_hours_mean.filna(np.mean(x_train.online_hours_mean))
driver_not_in_training=set(test_df.driver_id).difference(set(train_data.driver_id))
id=[id for id,x in enumerate(x_test.driver_id) if x in list(driver_not_in_training)]

independant=[x for x in x_train.columns if x not in ['online_hours']]
dependant=['online_hours']


from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error,mean_absolute_error

#Baseline model
#Linear regression
model_linear=LinearRegression()
model_linear.fit(x_train[independant],x_train[dependant])
predict_linear=model_linear.predict(x_test[independant])
predict_linear[id]=0

mean_squared_error(x_test[dependant],predict_linear)


gbm_param = {'learning_rate':[0.02,0.025,0.03,.035], 
           'n_estimators':[100,125,150,156,160,170],
           'max_depth':[2,3,4,5,6,7]}

tuning=GridSearchCV(estimator =GradientBoostingRegressor(min_samples_split=2,
                                                         min_samples_leaf=1,
                                                         subsample=1.0,
                                                         max_features=None),
                                                         param_grid = gbm_param,
                                                         scoring='neg_mean_squared_error',
                                                         n_jobs=4,
                                                         iid=False,
                                                         cv=3,
                                                         verbose=2)
tuning.fit(x_train[independant],x_train[dependant])
print(tuning.best_params_)

model_gbm=GradientBoostingRegressor(n_estimators=156,learning_rate=0.03)
model_gbm.fit(x_train[independant],x_train[dependant])
predict_gbm=model_gbm.predict(x_test[independant])
predict_gbm[id]=0
mean_squared_error(x_test[dependant],predict_gbm)


rf_param={'max_depth':[2,6,8,10,12,14,16,18,20]}
tuning_rf=GridSearchCV(estimator =RandomForestRegressor(min_samples_split=2,
                                                         min_samples_leaf=1,
                                                         bootstrap=True,
                                                         max_features=None),
                                                         param_grid = rf_param,
                                                         scoring='neg_mean_squared_error',
                                                         n_jobs=4,
                                                         iid=False,
                                                         cv=3,
                                                         verbose=2)
tuning_rf.fit(x_train[independant],x_train[dependant])
print(tuning_rf.best_params_)

model_rf=RandomForestRegressor(500,max_depth=10)
model_rf.fit(x_train[independant],x_train[dependant])
predict_rf=model_rf.predict(x_test[independant])
predict_rf[id]=0#replacing the predicted value of unsenn driver id with 0
mean_squared_error(x_test[dependant],predict_rf)

#x_train=x_train.drop(['mean_gender_hours'],axis=1)

#combining GBM and RF score
final=(predict_rf+
       predict_gbm)/2

mean_squared_error(x_test[dependant],final)

