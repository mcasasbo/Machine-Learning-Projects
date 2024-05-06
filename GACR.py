import sys
import os
import matplotlib
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd

import seaborn as sns
pd.options.display.max_rows = 100
plt.style.use('ggplot')
import matplotlib.pyplot as plt

from sklearn import model_selection
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, FunctionTransformer
from sklearn import set_config
from scipy import stats
from datetime import datetime

import random
import json
from pandas import json_normalize

from xml.etree.ElementInclude import include

# DATA IMPORT

columns = ['device', 'geoNetwork', 'totals'] # Columns with json format

p = 0.1 # Fraction of data to use

def json_read(df):
    data_frame = fire_dir + df
    
    df = pd.read_csv(data_frame, 
                     #converts json object into something readable by Python (whatever is in "dict" is a column)
                     converters={column: json.loads for column in columns}, 
                     dtype={'fullVisitorId': 'str'}, 
                     skiprows=lambda i: i>0 and random.random() > p)
    
    for column in columns: 
      #converts each column
        column_as_df = json_normalize(df[column]) 
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns] 
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
        
    print(f"Loaded {os.path.basename(data_frame)}. Shape: {df.shape}")
    return df

fire_dir = r'C:\Users\Usuario\Desktop\Proyects\Machine Learning\data'
file_name = r'\GACR_def.csv'

data_frame = fire_dir + file_name
p = 0.1 # Fracción de datos a utilizar
RANDOM_STATE= 42
random.seed(RANDOM_STATE)
df = json_read(file_name)

# DATA CLEANING

def data_preparation(df):
    df.set_index('sessionId', inplace = True)
    df_ids = ['fullVisitorId', 'visitId']
    df.drop(df_ids, axis = 1, inplace =True)
    constant_columns = [column for column in df.columns if df[column].nunique(dropna = False) == 1]
    df.drop(constant_columns, axis = 1, inplace = True)
    return df

df = data_preparation(df)

def target_prep(df):
    TARGET = 'totals.transactionRevenue'
    df[TARGET]= df[TARGET].astype(float)
    df[TARGET].fillna(0.0, inplace =  True)
    df[TARGET] =df[TARGET]/1000000
    df['visitWithTransaction'] = (df[TARGET]>0).astype(int)
    df['totals.transactionRevenueLOG'] = df[TARGET].apply(lambda x: np.log1p(x))
    TARGET_LOG = 'totals.transactionRevenueLOG'
    return df

df = target_prep(df)

TARGET = 'totals.transactionRevenue'
TARGET_LOG = 'totals.transactionRevenueLOG'
 

set_config(transform_output = "pandas")

null_imputer = ColumnTransformer(
    transformers=[
        ('imputezero', SimpleImputer(strategy='constant', fill_value='0'), ['totals.bounces', 'totals.newVisits']),
        ('pageviews_imputer', SimpleImputer(strategy='most_frequent'), ['totals.pageviews'])
    ]
)
df[['totals.bounces', 'totals.newVisits', 'totals.pageviews']] = null_imputer.fit_transform(df)

def date_prep(df, col_date, col_hour):
    df[col_date] = pd.to_datetime(df[col_date], format = '%Y%m%d')
    df['year'] = df[col_date].dt.year
    df['month'] = df[col_date].dt.month
    df['monthDay'] = df[col_date].dt.day
    df['weekDay'] = df[col_date].dt.weekday
    df['quarter'] = df[col_date].dt.quarter
    df['week'] = df[col_date].dt.week
    df['visitHour'] = df[col_hour].apply(lambda x: datetime.fromtimestamp(x).hour)
    cols_to_drop = [col_date, col_hour]
    return df.drop(cols_to_drop, axis = 1, inplace = True)

date_prep(df, 'date', 'visitStartTime')


cols_to_int = ['device.isMobile', 'totals.hits', 'totals.pageviews']
df[cols_to_int] = df[cols_to_int].astype(int)


def setOthers(dataframe, column, num_values):
    top_categories = dataframe[column].value_counts().head(num_values)
    top_categories_list = top_categories.index.to_list()
    top_categories_list.append('Others')
    dataframe[column] = pd.Categorical(dataframe[column], categories=top_categories_list)
    return dataframe[column].fillna('Others')

df['device.browser'] = setOthers(df,'device.browser',5)
df['device.operatingSystem'] = setOthers(df,'device.operatingSystem',5)

df['network_net'] = df['geoNetwork.networkDomain'].str.contains('.net', case=False).astype(int)
df['network_com'] = df['geoNetwork.networkDomain'].str.contains('.com', case=False).astype(int)
df['network_unknown'] = df['geoNetwork.networkDomain'].str.contains('(not set)|unknown').astype(int)
df['other_networks'] = ~df['geoNetwork.networkDomain'].str.contains('.com|.net | (not set)|unknown', case=False)
df.drop('geoNetwork.networkDomain', axis = 1, inplace = True)

def explore_cat_values(dataframe, column, target_column):
    _results_df = dataframe[dataframe[target_column] > 1].pivot_table(index=column, values=target_column, aggfunc=[len, np.mean])
    _results_df.columns = ['transactions', 'mean_revenue_ln']
    _results_df['n_rows'] = dataframe[column].value_counts(dropna=False)
    _results_df['pct_rows'] = dataframe[column].value_counts(normalize=True, dropna=False).round(3)
    _results_df['pct_transactions'] = (_results_df['transactions'] / _results_df['n_rows']).round(3)
    _results_df = _results_df[['n_rows', 'pct_rows', 'transactions', 'pct_transactions', 'mean_revenue_ln']]
    return _results_df

def setOthersminmax(dataframe, column, target_column, num_rows_min, top_n):
    results_by_category = explore_cat_values(dataframe, column, target_column)
    last_categories = results_by_category[results_by_category['n_rows'] > num_rows_min].sort_values(by='pct_transactions').head(top_n).index.to_list()
    first_categories = results_by_category[results_by_category['n_rows'] > num_rows_min].sort_values(by='pct_transactions').tail(top_n).index.to_list()
    top_categories_list = list(set(first_categories + last_categories))
    top_categories_list.append('Others')
    dataframe[column] = pd.Categorical(dataframe[column], categories=top_categories_list)
    return dataframe[column].fillna('Others')

df['geoNetwork.country'] = setOthersminmax(df, 'geoNetwork.country', TARGET_LOG, 500, 5)
df['geoNetwork.city'] = setOthersminmax(df, 'geoNetwork.city', TARGET_LOG, 500, 10)


df.drop(['geoNetwork.region', 'geoNetwork.metro', 'totals.pageviews'], axis = 1, inplace = True)

df['totals.hits_LOG'] = df['totals.hits'].apply(lambda x: np.log1p(x))
df['visitNumber_LOG']= df['visitNumber'].apply(lambda x: np.log1p(x))


# DATA PREPROCESSING

cat_OHE = ['channelGrouping','device.browser','device.operatingSystem','device.deviceCategory', 'geoNetwork.continent']
cat_to_int = ['totals.bounces', 'totals.newVisits', 'other_networks']
cat_freq = ['geoNetwork.subContinent', 'geoNetwork.country','geoNetwork.city']
cat_to_transform = ['channelGrouping', 'device.browser', 'device.operatingSystem', 'device.deviceCategory',
 'geoNetwork.continent', 'geoNetwork.subContinent', 'geoNetwork.country', 'geoNetwork.city', 'totals.bounces',
 'totals.newVisits', 'other_networks']

transform_cat = ColumnTransformer(transformers = [
    ('encoder', OneHotEncoder(sparse_output = False), cat_OHE),
    ('dtype_conv', FunctionTransformer(lambda x: x.astype(float)), cat_to_int),
    ('ordinal', OrdinalEncoder(),  cat_freq)
])

df1 = transform_cat.fit_transform(df)
df = df.merge(df1, left_index = True, right_index = True)
df.drop(cat_to_transform, axis = 1, inplace = True)

cols_to_drop = ['visitNumber', 'totals.hits','visitWithTransaction', 'totals.transactionRevenue']
df = df.drop(cols_to_drop, axis = 1)

# MODELLING

## Validation strategy
def validation_strategy(df):
    df_val = df[df['year'] * 100 + df['month'] >= 201706]
    df_dev = df[df['year'] * 100 + df['month'] < 201706]

    df_val_X = df_val.drop(TARGET_LOG, axis = 1)
    df_val_y = df_val[[TARGET_LOG]]
    df_dev_X = df_dev.drop(TARGET_LOG, axis = 1)
    df_dev_y = df_dev[[TARGET_LOG]]
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
    df_dev_X,
    df_dev_y,
    random_state =  42,
    test_size = 0.25)
    return df

validation_strategy(df)


# XGB MODELLING
def xgb_model(df):
    validation_strategy(df)
    xgb_model = xgb.XGBRegressor(eval_metric = "rmse", early_stopping_rounds = 10, random_state = 42, n_estimators = 100, max_depth = 4)
    xgb_model.fit(X_train, y_train, 
        eval_set = [(X_train, y_train), (df_val_X, df_val_y)], 
        verbose = True)
    Y_train_predict = xgb_model.predict(X_train)
    Y_valida_predict = xgb_model.predict(df_val_X)






#data_preparation = ColumnTransformer(
#    transformers = [
#        ('device.browser_set', setOthers(df,'device.browser',5)),
#        ('device.operatingSystem_set', setOthers(df,'device.operatingSystem',6)),
#        ('geoNetwork.networkDomain_set', setOthers(df,'geoNetwork.networkDomain',10)),
#        ('country_set', setOthersminmax(df, 'geoNetwork.country', TARGET, 500, 5)),
#        ('city_set', setOthersminmax(df, 'geoNetwork.city', TARGET, 100, 10)),
#        ('date', date_transformer, ['date'])