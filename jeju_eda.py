# -*- coding: utf-8 -*-
"""제주도 특산물 EDA.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1WHautKJkxdeeAKLGNbt6Ta3Z7WqwvxMB
"""

pip install catboost

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier, Pool, cv
import catboost
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import log_loss

import pandas as pd
data = pd.read_csv('/content/train.csv')
data.head()

data['item'].value_counts()

data.isna().sum()

type(data['timestamp'][0])

import datetime
data['timestamp'] = pd.to_datetime(data['timestamp'])

type(data['timestamp'][0])

import seaborn as sns
import matplotlib.pyplot as plt
sns.boxplot(x='item',y='price(??kg)',data=data)

sns.boxplot(x='corporation',y='price(??kg)',data=data)

sns.boxplot(x='location',y='price(??kg)',data=data)

plt.figure(figsize=(10,8))
sns.scatterplot(data=data,x ='supply(kg)',y='price(??kg)')

plt.figure(figsize=(10,8))
sns.scatterplot(data=data,x ='supply(kg)',y='price(??kg)',hue='item')

plt.figure(figsize=(10,8))
sns.scatterplot(data=data,x ='supply(kg)',y='price(??kg)',hue='corporation')

plt.figure(figsize=(10,8))
sns.scatterplot(data=data,x ='supply(kg)',y='price(??kg)',hue='location')

data[['supply(kg)','price(??kg)']].corr()

data['year'] = data['timestamp'].dt.year
data['month'] = data['timestamp'].dt.month
data['day'] = data['timestamp'].dt.day
data

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

sns.barplot(data=data,x='year',y='price(??kg)',ax=axes[0],palette='husl')
sns.barplot(data=data,x='month',y='price(??kg)',ax=axes[1])

items = data['item'].value_counts().index.to_list()
items

fig, axes = plt.subplots(5,1,figsize=(10,15))
for i in range(len(items)):
    r = i%5
    sns.scatterplot(data = data[data['item']==items[i]], x = 'supply(kg)',y='price(??kg)',ax=axes[r],label=items[i])

data[data['supply(kg)']==0]

len(data) , len(data[data['supply(kg)']==0])

len(data[data['supply(kg)']==0]) / len(data)

sup_notzero = data[data['supply(kg)']!=0]
sup_notzero

fig, axes = plt.subplots(5,1,figsize=(10,15))
for i in range(len(items)):
    r = i%5
    sns.scatterplot(data = sup_notzero[sup_notzero['item']==items[i]], x = 'supply(kg)',y='price(??kg)',ax=axes[r],label=items[i])

sup_notzero['y-m']=sup_notzero['timestamp'].dt.strftime('%Y-%m')
sup_notzero['y-m']

plt.figure(figsize=(15,8))
plt.plot(sup_notzero[sup_notzero['item'] == 'TG'].groupby(['y-m'])['price(??kg)'].mean())

plt.figure(figsize=(15,8))
plt.plot(sup_notzero[sup_notzero['item'] == 'BC'].groupby(['y-m'])['price(??kg)'].mean())

plt.figure(figsize=(15,8))
plt.plot(sup_notzero[sup_notzero['item'] == 'RD'].groupby(['y-m'])['price(??kg)'].mean())

plt.figure(figsize=(15,8))
plt.plot(sup_notzero[sup_notzero['item'] == 'CB'].groupby(['y-m'])['price(??kg)'].mean())

plt.figure(figsize=(15,8))
plt.plot(sup_notzero[sup_notzero['item'] == 'CR'].groupby(['y-m'])['price(??kg)'].mean())

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

sns.heatmap(sup_notzero.corr(),annot=True,cmap='Reds',ax = axes[0])
axes[0].set_title('supply_NOT_zero')
sns.heatmap(data.corr(),annot=True,cmap='Reds',ax = axes[1])
axes[1].set_title('all')

corp = data['corporation'].value_counts().index.to_list()
corp

fig, axes = plt.subplots(2,3,figsize=(10,10))
for i in range(len(corp)):
    n=int(i/3)
    r = i%3
    sns.scatterplot(data = sup_notzero[sup_notzero['corporation']==corp[i]], x = 'supply(kg)',y='price(??kg)',ax=axes[n][r],label=corp[i])

pip install pytimekr

from pytimekr import pytimekr

year_2019 = pytimekr.holidays(year=2019)
year_2020 = pytimekr.holidays(year=2020)
year_2021 = pytimekr.holidays(year=2021)
year_2022 = pytimekr.holidays(year=2022)
year_2023 = pytimekr.holidays(year=2023)



def holidays(x):
    if x.weekday() in range(5,8):
        return 1
    if x.year == 2019  and x in year_2019 :
        return 1
    elif x.year == 2020 and x in year_2020:
        return 1
    elif x.year == 2021 and x in year_2021 :
        return 1
    elif x.year == 2022 and x in year_2022 :
        return 1
    elif x.year == 2023 and x in year_2023:
        return 1
    else:
        return 0

import warnings
warnings.filterwarnings('ignore')
data['holiday'] = data['timestamp'].apply(holidays)
data

data['holiday'].value_counts()

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

sns.barplot(data=data,x='item',y='price(??kg)',hue='holiday',ax=axes[0],palette='husl')
sns.barplot(data=data,x='corporation',y='price(??kg)',hue='holiday',ax=axes[1],palette='husl')
sns.barplot(data=data,x='location',y='price(??kg)',hue='holiday',ax=axes[2],palette='husl')

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

sns.barplot(data=data,x='item',y='supply(kg)',hue='holiday',ax=axes[0],palette='husl')
sns.barplot(data=data,x='corporation',y='supply(kg)',hue='holiday',ax=axes[1],palette='husl')
sns.barplot(data=data,x='location',y='supply(kg)',hue='holiday',ax=axes[2],palette='husl')

data.columns

df1= pd.pivot_table(data,index='item',values='price(??kg)',aggfunc='sum')
df1['비율'] = (df1['price(??kg)']/df1['price(??kg)'].sum())*100
df1

plt.figure(figsize=(10,8))
colors = ['red','yellow','purple','goldenrod','lightcoral']
plt.rc('font',family='Malgun Gothic')
# TG : 감귤 BC : 브로콜리 RD : 무 CR : 당근 CB : 양배추
plt.pie(data = df1,x = 'price(??kg)',labels=df1.index,autopct=lambda x : '{:.1f}%'.format(x),colors=colors)
plt.legend()
plt.show()

df2 = pd.pivot_table(data,index='corporation',values='price(??kg)',aggfunc='sum')
df2['비율'] = (df2['price(??kg)']/df2['price(??kg)'].sum())*100
df2

plt.figure(figsize=(10,8))
colors = ['red','yellow','purple','goldenrod','lightcoral','pink']
plt.rc('font',family='Malgun Gothic')
# TG : 감귤 BC : 브로콜리 RD : 무 CR : 당근 CB : 양배추
plt.pie(data = df2,x = 'price(??kg)',labels=df2.index,autopct=lambda x : '{:.1f}%'.format(x),colors=colors)
plt.legend()
plt.show()



plt.figure(figsize=(10,8))
sns.distplot(data['price(??kg)'])

plt.figure(figsize=(10,8))
sns.distplot(sup_notzero['price(??kg)'])

sup_notzero['price(??kg)'].skew(), data['price(??kg)'].skew()

sup_notzero['price(??kg)'].kurt(), data['price(??kg)'].kurt()

sns.boxplot(y='price(??kg)',data=data)

sns.boxplot(y='price(??kg)',data=sup_notzero)

data.columns

# 이상치제거함수
from collections import Counter
import numpy as np
def outlier(df,n,cols):
    outs = []
    for col in cols :
        Q1 = np.percentile(df[col],25)
        Q3 = np.percentile(df[col],75)
        IQR = Q3 - Q1

        step = 1.5*IQR
        indexes = df[(df[col] < Q1 - step)|(df[col] > Q3 + step)].index
        outs.extend(indexes)
    outs = Counter(outs)
    res = [k for k,v in outs.items() if v > n]
    return res

outlier_col = ['supply(kg)','price(??kg)']
outlier(data,2,outlier_col)

outlier(sup_notzero,2,outlier_col)

data['supply(kg)'].value_counts()