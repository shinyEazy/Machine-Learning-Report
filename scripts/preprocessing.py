import numpy as np
import pandas as pd
import os
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import VotingRegressor
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('darkgrid')
from sklearn.metrics import make_scorer, cohen_kappa_score
import warnings
warnings.simplefilter('ignore')
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from IPython.display import clear_output
from scipy.optimize import minimize
from colorama import Fore, Style
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Reads a parquet file, drops the 'step' column, and returns stats and ID
def process_file(filename, dirname):
    df = pd.read_parquet(os.path.join(dirname, filename, 'part-0.parquet'))
    df.drop('step', axis=1, inplace=True)
    return df.describe().values.reshape(-1), filename.split('=')[1]

# Loads time-series data, processes each file, and compiles into a DataFrame
def load_time_series(dirname) -> pd.DataFrame:
    ids = os.listdir(dirname)
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(lambda fname: process_file(fname, dirname), ids), total=len(ids)))

    stats, indexes = zip(*results)
    df = pd.DataFrame(stats, columns=[f'stat_{i}' for i in range(len(stats[0]))])
    df['id'] = indexes
    return df

# Define a function to replace NaN with 'Missing' and change the type of season columns to 'category'
def update(df):
    global season_cat
    for cat in season_cat:
        df[cat] = df[cat].fillna('Missing')  # Replace NaN values with 'Missing'
        df[cat] = df[cat].astype('category')  # Convert column to 'category' type to optimize storage
    return df

# Define a function to create a mapping of category values to numbers
def create_mapping(column, dataset):
    unique_values = dataset[column].unique()  # Get unique category values from the column
    return {value: idx for idx, value in enumerate(unique_values)}  # Create a mapping of category to number

# Load static data
train = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/train.csv')
test = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/test.csv')
sample = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/sample_submission.csv')

# Load time series data for train and test
train_ts = load_time_series("/kaggle/input/child-mind-institute-problematic-internet-use/series_train.parquet")
test_ts = load_time_series("/kaggle/input/child-mind-institute-problematic-internet-use/series_test.parquet")

# Extract time-series feature columns by removing 'id' from train_ts columns
time_series_cols = train_ts.columns.tolist()
time_series_cols.remove("id")

# Merge time-series data with train and test datasets on 'id'
train = pd.merge(train, train_ts, how="left", on='id')
test = pd.merge(test, test_ts, how="left", on='id')

# Drop the 'id' column from both train and test datasets
train = train.drop('id', axis=1)
test = test.drop('id', axis=1)

# Define a list of 58 features (excluding 'id' and including 'sii') that appear in both train and test datasets
featuresCols = ['Basic_Demos-Enroll_Season', 'Basic_Demos-Age', 'Basic_Demos-Sex', 
                'CGAS-Season', 'CGAS-CGAS_Score', 
                'Physical-Season', 'Physical-BMI', 'Physical-Height', 
                'Physical-Weight', 'Physical-Waist_Circumference',
                'Physical-Diastolic_BP', 'Physical-HeartRate', 'Physical-Systolic_BP',
                'Fitness_Endurance-Season', 'Fitness_Endurance-Max_Stage',
                'Fitness_Endurance-Time_Mins', 'Fitness_Endurance-Time_Sec',
                'FGC-Season', 'FGC-FGC_CU', 'FGC-FGC_CU_Zone', 'FGC-FGC_GSND',
                'FGC-FGC_GSND_Zone', 'FGC-FGC_GSD', 'FGC-FGC_GSD_Zone', 'FGC-FGC_PU',
                'FGC-FGC_PU_Zone', 'FGC-FGC_SRL', 'FGC-FGC_SRL_Zone', 'FGC-FGC_SRR',
                'FGC-FGC_SRR_Zone', 'FGC-FGC_TL', 'FGC-FGC_TL_Zone', 
                'BIA-Season', 'BIA-BIA_Activity_Level_num', 'BIA-BIA_BMC', 'BIA-BIA_BMI', 
                'BIA-BIA_BMR', 'BIA-BIA_DEE', 'BIA-BIA_ECW', 'BIA-BIA_FFM', 
                'BIA-BIA_FFMI', 'BIA-BIA_FMI', 'BIA-BIA_Fat', 'BIA-BIA_Frame_num', 
                'BIA-BIA_ICW', 'BIA-BIA_LDM', 'BIA-BIA_LST', 'BIA-BIA_SMM', 'BIA-BIA_TBW', 
                'PAQ_A-Season', 'PAQ_A-PAQ_A_Total', 'PAQ_C-Season', 'PAQ_C-PAQ_C_Total', 
                'SDS-Season', 'SDS-SDS_Total_Raw', 'SDS-SDS_Total_T', 
                'PreInt_EduHx-Season', 'PreInt_EduHx-computerinternet_hoursday', 
                'sii']

# Add the time series feature columns to the list of features
featuresCols += time_series_cols

# Select the specified features from the train dataset
train = train[featuresCols]

# Remove rows where the 'sii' column has missing values
train = train.dropna(subset='sii')

# Define a list of categorical features that represent season data
season_cat = ['Basic_Demos-Enroll_Season', 'CGAS-Season', 'Physical-Season', 
              'Fitness_Endurance-Season', 'FGC-Season', 'BIA-Season', 
              'PAQ_A-Season', 'PAQ_C-Season', 'SDS-Season', 'PreInt_EduHx-Season']

# Apply the update function to the train and test datasets to handle missing values and optimize column types
train = update(train)
test = update(test)

# Label encoding for season columns
for cat in season_cat: 
    # Create mapping for train and test datasets
    mapping_train = create_mapping(cat, train)
    mapping_test = create_mapping(cat, test)

    # Replace the categorical values with their corresponding numeric encoding and convert to integer data type
    train[cat] = train[cat].replace(mapping_train).astype(int)
    test[cat] = test[cat].replace(mapping_test).astype(int)