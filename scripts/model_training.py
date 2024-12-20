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

# Define a function to calculate the quadratic weighted kappa score
def quadratic_weighted_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

# Define a function to round values based on thresholds
def threshold_rounder(oof_non_rounded, thresholds):
    # Apply thresholds to classify values into categories: 0, 1, 2, or 3
    return np.where(oof_non_rounded < thresholds[0], 0,
                   np.where(oof_non_rounded < thresholds[1], 1,
                           np.where(oof_non_rounded < thresholds[2], 2, 3)))

# Define a function to train the model, perform cross-validation, and evaluate using QWK
def TrainML(model_class, test_data):
    # Split data into features (X) and target (y)
    X = train.drop(['sii'], axis=1)
    y = train['sii']

    # Define StratifiedKFold for cross-validation
    SFK = StratifiedKFold(n_splits=5, shuffle=True, random_state=12)

    train_S = []  # Store QWK scores for training
    test_S = []   # Store QWK scores for validation

    oof_non_rounded = np.zeros(len(y), dtype=float)  # Out-of-fold predictions (non-rounded)
    oof_rounded = np.zeros(len(y), dtype=int)        # Out-of-fold predictions (rounded)
    test_preds = np.zeros((len(test_data), 5))        # Store test predictions for each fold

    # Loop over each fold for cross-validation
    for fold, (train_idx, test_idx) in enumerate(tqdm(SFK.split(X, y), desc="Training Folds", total=5)):
        X_train, X_val = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[test_idx]

        # Clone and train the model
        model = clone(model_class)
        model.fit(X_train, y_train)

        # Make predictions on training and validation data
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

        # Store out-of-fold predictions
        oof_non_rounded[test_idx] = y_val_pred
        y_val_pred_rounded = y_val_pred.round(0).astype(int)
        oof_rounded[test_idx] = y_val_pred_rounded

        # Calculate QWK for training and validation
        train_kappa = quadratic_weighted_kappa(y_train, y_train_pred.round(0).astype(int))
        val_kappa = quadratic_weighted_kappa(y_val, y_val_pred_rounded)

        train_S.append(train_kappa)
        test_S.append(val_kappa)

        # Store test predictions
        test_preds[:, fold] = model.predict(test_data)

        # Print fold results
        print(f"Fold {fold+1} - Train QWK: {train_kappa:.4f}, Validation QWK: {val_kappa:.4f}")
        clear_output(wait=True)

    # Print average QWK scores
    print(f"Mean Train QWK --> {np.mean(train_S):.4f}")
    print(f"Mean Validation QWK ---> {np.mean(test_S):.4f}")

    # Optimize the threshold for better QWK performance
    KappaOPtimizer = minimize(evaluate_predictions,
                              x0=[0.5, 1.5, 2.5], args=(y, oof_non_rounded), 
                              method='Nelder-Mead')
    
    assert KappaOPtimizer.success, "Optimization did not converge."
    
    # Apply optimized thresholds to tune predictions
    oof_tuned = threshold_rounder(oof_non_rounded, KappaOPtimizer.x)
    tKappa = quadratic_weighted_kappa(y, oof_tuned)

    print(f"----> || Optimized QWK SCORE :: {Fore.CYAN}{Style.BRIGHT} {tKappa:.3f}{Style.RESET_ALL}")

    # Average test predictions and apply tuned threshold
    tpm = test_preds.mean(axis=1)
    tpTuned = threshold_rounder(tpm, KappaOPtimizer.x)
    
    # Create submission DataFrame
    submission = pd.DataFrame({
        'id': sample['id'],
        'sii': tpTuned
    })

    return submission