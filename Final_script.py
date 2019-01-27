#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 17:03:10 2019

@author: s0p00zp
"""

import pandas as pd
import numpy as np
import datetime
import time
import sys
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error

#Specifying the file details
path='/Users/s0p00zp/.kaggle/DS TakeHome Assignment/data'
out_path='/Users/s0p00zp/.kaggle/DS TakeHome Assignment/data'
pings='pings.csv'
driver='drivers.csv'
test='test.csv'

pings_df=pd.read_csv(path+'/'+pings)
drivers_df=pd.read_csv(path+'/'+driver)
test_df=pd.read_csv(path+'/'+test)

def Extracting_Date(pings_df):
    pings_df['date']=[time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime(int(x)+25200)) for x in pings_df.ping_timestamp]
    pings_df['date']=pd.to_datetime(pings_df['date'])
    pings_df['hour']=pings_df['date'].dt.hour
    pings_df['date_month']=pings_df['date'].dt.date
    return(pings_df)
    
pings_df=Extracting_Date(pings_df)



duplicate=pings_df.duplicated()
print("number of duplicate records:",np.sum(duplicate))

print("removing duplicate records")
pings_df_clean=pings_df[~duplicate]

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

train_df_mean=train_data.groupby(['driver_id']).agg(f)
train_df_mean=train_df_mean.reset_index()
train_df_mean.columns=train_df_mean.columns.map('_'.join)
train_df_mean=train_df_mean.rename(columns={"driver_id_":"driver_id"})

all_data_M=pd.merge(train_data.drop_duplicates(),drivers_df.drop_duplicates(),how='inner',on='driver_id')

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
    x_train=x[x.ind=='tr']
    x_test=x[x.ind=='te']
    x_train=x_train.drop(['ind'],axis=1)
    x_test=x_test.drop(['ind'],axis=1)
    return(x_train,x_test)

x_train,x_test=Creating_train_test(train_data,test_df,drivers_df,label_encode=1,merge=0)
x_train['key']=x_train['driver_id'].astype('str')+"_"+x_train['date'].astype('str')
x_train=x_train.drop_duplicates(subset='key')
x_test['key']=x_test['driver_id'].astype('str')+"_"+x_test['date'].astype('str')
x_test=x_test.drop_duplicates(subset='key')
x_test=x_test.reset_index()
test_df['key']=test_df['driver_id'].astype('str')+"_"+test_df['date'].astype('str')
test_duplicates=test_df[test_df.duplicated(subset='key')]
test_df=test_df.drop_duplicates(subset='key')


driver_not_in_training=set(test_df.driver_id).difference(set(train_data.driver_id))
id=[id for id,x in enumerate(x_test.driver_id) if x in list(driver_not_in_training)]

independant=[x for x in x_train.columns if x not in ['online_hours','date','key']]
dependant=['online_hours']

model_gbm=GradientBoostingRegressor(n_estimators=156,learning_rate=0.03)
model_gbm.fit(x_train[independant],x_train[dependant])
predict_gbm=model_gbm.predict(x_test[independant])
predict_gbm[id]=0


model_rf=RandomForestRegressor(500,max_depth=10)
model_rf.fit(x_train[independant],x_train[dependant])
predict_rf=model_rf.predict(x_test[independant])
predict_rf[id]=0

final=(predict_rf+
       predict_gbm)/2
       
test_df['predicted_hours']=final
test_df=test_df[['driver_id','date','online_hours''predicted_hours']]
test_df.to_csv(out_path+"/"+"predicted_result.csv",index=False)









