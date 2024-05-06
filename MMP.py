# Imports
    ## Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn import set_config
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, MinMaxScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, roc_auc_score
from sklearn import model_selection
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, RepeatedKFold, KFold
import category_encoders as ce
import folium
import plotly.express as px
from scipy import stats

    ## Data
file_dir = r'C:\Users\Usuario\Desktop\Proyects\Machine Learning\data'
file_name =  r"\sample_mmp.csv"
df = pd.read_csv(file_dir + file_name, sep = ",")
n_filas = int(len(df)*1)
df = df.sample(n = n_filas, random_state = 42)

# Functions
def var_low_variance(dataframe):
    low_variance_cols = []
    unbalanced_cols = []
    
    for column in dataframe.columns:
        unique_values = dataframe[column].unique()
        if len(unique_values) <= 2:  # Variables binarias o categóricas con 2 valores únicos
            value_counts = dataframe[column].value_counts(normalize=True, dropna=False)
            if len(value_counts) == 1 or (len(value_counts) == 2 and value_counts.max() >= 0.95):
                low_variance_cols.append(column)
            elif len(value_counts) == 2 and value_counts.max() >= 0.95:
                unbalanced_cols.append(column)
        else:  # Variables categóricas con más de 2 valores únicos
            value_counts = dataframe[column].value_counts(normalize=True, dropna=False)
            if value_counts.max() >= 0.95:
                unbalanced_cols.append(column)
    
    return low_variance_cols, unbalanced_cols

def bolean_var(dataframe):
    bolean_vars = []
    for var in dataframe.columns:
        if dataframe[var].nunique() == 2:
            bolean_vars.append(var)
    return bolean_vars

def category_lists(dataframe, list):
    cols_to_OHE = []
    cols_to_ordinal = []
    cols_to_mf = []
    for c in dataframe[list]:
        n_cat = dataframe[c].nunique()
        if 5<= n_cat <=20:
            cols_to_OHE.append(c)
        elif n_cat > 20:
            cols_to_ordinal.append(c)
        else:
            cols_to_mf.append(c)
    return cols_to_OHE, cols_to_ordinal, cols_to_mf

def numeric_lists_transf(df, num_cols):
    minmaxscaler_vars = []
    log_vars = []
    
    summary = df.describe()
    IQR = summary.loc['75%'] - summary.loc['25%']
    
    lower_bound = summary.loc['25%'] - 1.5 * IQR
    upper_bound = summary.loc['75%'] + 1.5 * IQR
    
    for col in num_cols:
        outliers_lower = df[col] < lower_bound[col]
        outliers_upper = df[col] > upper_bound[col]
        
        if outliers_lower.any() or outliers_upper.any():
            log_vars.append(col)
        else:
            minmaxscaler_vars.append(col)

    return log_vars, minmaxscaler_vars

def convert_cat_bool(df, cols):
    for col in cols:
        moda_category = df[col].value_counts().idxmax()
        prefix = str(moda_category)
        new_column_name = f"{col}_{prefix}"
        df[new_column_name] = df[col].apply(lambda x: 1 if x == prefix else 0)
        df.drop(columns=col, inplace=True)
    return df

def validation_strategy_cl(dataframe, TARGET):
    X_train_, X_val, y_train_, y_val = train_test_split(dataframe.drop(TARGET, axis=1), dataframe[TARGET], test_size=0.15, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_train_, y_train_, test_size = 0.20, random_state = 42)
    X_train_, X_val, y_train_, y_val = train_test_split(dataframe.drop(TARGET, axis=1), dataframe[TARGET], test_size=0.15, random_state=42)
    return X_train, X_test, y_train, y_test, X_val, y_val

# Preprocessing
set_config(transform_output = "pandas")
df = df.set_index('MachineIdentifier')
df.drop('Unnamed: 0', axis = 1, inplace = True)

TARGET = ['HasDetections']

low_variance_cols, unbalanced_cols = var_low_variance(df)
df_c = df.drop(low_variance_cols + unbalanced_cols + ['Census_InternalPrimaryDisplayResolutionHorizontal', 
                                                      'Census_InternalPrimaryDisplayResolutionVertical'], axis =1)
cols_to_obj = ['AVProductsInstalled', 'CountryIdentifier', 'CityIdentifier', 'OrganizationIdentifier', 'GeoNameIdentifier', 
 'LocaleEnglishNameIdentifier', 'OsBuild', 'OsSuite', 'IeVerIdentifier', 'Census_OEMNameIdentifier', 'Census_ProcessorCoreCount',
'Census_ProcessorManufacturerIdentifier', 'Census_ProcessorModelIdentifier', 'Census_InternalBatteryNumberOfCharges',
 'Census_OSBuildNumber', 'Census_OSBuildRevision', 'Census_OSInstallLanguageIdentifier', 'Census_FirmwareManufacturerIdentifier',
   'Wdft_RegionIdentifier']

df_c[cols_to_obj] = df_c[cols_to_obj].astype(str)

boolean_columns = bolean_var(df_c)
boolean_columns = [x for x in boolean_columns if x not in TARGET]
numeric_columns = df_c.select_dtypes(include = 'number').drop(TARGET, axis = 1).columns
numeric_columns = [x for x in numeric_columns if x not in boolean_columns]
object_columns = df_c.select_dtypes(include = ['object', 'string']).columns
cols_constant_1 = ['Census_ThresholdOptIn', 'Census_IsFlightingInternal', 'Census_IsWIMBootEnabled']
boolean_columns = [x for x in boolean_columns if x not in cols_constant_1]
numeric_columns = [x for x in numeric_columns if x not in cols_constant_1]

impute_pipe = ColumnTransformer(transformers = [
    ("median", SimpleImputer(strategy = "median"), numeric_columns),
    ("mf", SimpleImputer(strategy = "most_frequent"), object_columns),
    ("mf_b", SimpleImputer(strategy = "most_frequent"), boolean_columns),
    ("lf", SimpleImputer(strategy = "constant", fill_value= 1), cols_constant_1)
], remainder = 'passthrough')
df_c = impute_pipe.fit_transform(df_c)

df_t = df_c
TARGET = ['remainder__HasDetections']

boolean_columns = bolean_var(df_t)
boolean_columns = [x for x in boolean_columns if x not in TARGET]
numeric_columns = df_c.select_dtypes(include = 'number').drop(TARGET, axis = 1).columns
numeric_columns = [x for x in numeric_columns if x not in boolean_columns]
object_columns = df_c.select_dtypes(include = ['object', 'string']).columns

cols_to_OHE, cols_to_ordinal, cols_to_mf = category_lists(df_t, object_columns)
log_vars, minmaxscaler_vars = numeric_lists_transf(df_t, numeric_columns)

convert_bol_t = FunctionTransformer(func=lambda X: convert_cat_bool(X, cols=cols_to_mf), validate=False)
log_trans = FunctionTransformer(func=lambda X: np.log(X))

transform_pipe = ColumnTransformer(transformers=[
    ("scaler", MinMaxScaler(), minmaxscaler_vars),
    ("log", log_trans, log_vars),
    ("encoder", OneHotEncoder(sparse_output=False), cols_to_OHE),
    ("ordinal", OrdinalEncoder(), cols_to_ordinal),
    ('bool', convert_bol_t, cols_to_mf)
], remainder='passthrough')
df_t2 = transform_pipe.fit_transform(df_t)

# Modelling
TARGET = ['remainder__remainder__HasDetections']
X_train, X_test, y_train, y_test, X_val, y_val = validation_strategy_cl(df_t2, TARGET)

gb = GradientBoostingClassifier(n_estimators=50, max_depth=5, min_samples_split=500, random_state=42)
gb.fit(X_train, np.ravel(y_train))