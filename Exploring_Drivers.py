#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 10:52:21 2019

@author: s0p00zp
"""

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import seaborn as sns
import datetime
import time
import matplotlib.pyplot as plt

#Specifying the file details
path='/Users/s0p00zp/.kaggle/DS TakeHome Assignment/data'
pings='pings.csv'
driver='drivers.csv'
test='test.csv'

pings_df=pd.read_csv(path+'/'+pings)
drivers_df=pd.read_csv(path+'/'+driver)
test_df=pd.read_csv(path+'/'+test)

########## Exploring Drivers data ############
drivers_df.head()


#checking number of unique driver ID's
print("number of unique driver ID's: \n",len(set(drivers_df.driver_id)))

#checking for missing values in data
print("missing values in data: \n",drivers_df.isna().any())

print("Rows/columns in data: \n",drivers_df.shape)

#Drivers Data types
print("Data types: \n",drivers_df.dtypes)

#Driver Summary
print("Driver Summary: \n",drivers_df.describe())

#exploring Drivers Data
plt.figure()
print(sns.countplot(drivers_df.gender).set_title('Gender wise count'))

plt.figure()
print(sns.boxplot(drivers_df.age).set_title('age distribution'))
plt.figure()
print(sns.boxplot(drivers_df.gender,drivers_df.age).set_title('gender vs age'))

print("75th percentile for female age:",drivers_df['age'][drivers_df['gender']=='FEMALE'].quantile(0.75))
print("90th percentile for female age:",drivers_df['age'][drivers_df['gender']=='FEMALE'].quantile(0.90))
print("99th percentile for female age:",drivers_df['age'][drivers_df['gender']=='FEMALE'].quantile(0.99))
print("high numbers for female age:\n",drivers_df['age'][(drivers_df['gender']=='FEMALE')&(drivers_df['age']>71)])



#Grouping the age into buckets and counts
print("Grouping the age into buckets and counts: \n",pd.cut(drivers_df.age,3,).value_counts())
print("#Higher the age of the drivers Lower is the number.\n","#Younger age drivers are more")

plt.figure()
print(sns.distplot(drivers_df.number_of_kids).set_title('number of Kids'))

print("Gender vs no of kids count: \n")
print(pd.crosstab(drivers_df.gender,drivers_df.number_of_kids))
