#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 11:32:05 2020.

@author: peter
"""
# Standard library imports
import pathlib

# Imports
import numpy as np
# import numpy.ma as ma
# import pandas as pd
import geopandas as gpd
from sklearn.model_selection import train_test_split  # ,KFold, cross_val_score
# from skopt.utils import use_named_args
from sklearn.preprocessing import MinMaxScaler
# from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score
from skopt.learning import RandomForestRegressor
from sklearn.metrics import mean_squared_error
# from skopt import gp_minimize, dump, load
# from skopt.space import Integer  # , Real
# from skopt.plots import plot_objective, plot_evaluations, plot_convergence
# from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt

# Custom imports
from feat_eng.funcs import add_min, safe_log  # , get_corr_feats, min_max
from custom_metrics.metrics import (mean_error, lin_ccc,
                                    model_efficiency_coefficient)


# ------------------- Settings ---------------------------------------------- #


# Set matploblib style
plt.style.use('seaborn-colorblind')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
mpl.rcParams['figure.dpi'] = 450
mpl.rcParams['savefig.transparent'] = True
mpl.rcParams['savefig.format'] = 'svg'


# # Reset params if needed
# mpl.rcParams.update(mpl.rcParamsDefault)


# ------------------- Organization ------------------------------------------ #


data_dir = pathlib.Path('data/')
optimizer_dir = pathlib.Path('optimizers/')
SEED = 43


# ------------------- Read and prep data ------------------------------------ #


# Load target data
target_data = gpd.read_file(data_dir.joinpath('germany_targets.geojson'),
                            driver='GeoJSON')
# Get target array
targets = target_data.OC.values
# Load feature array
features = np.load(data_dir.joinpath('numerical_feats.npy'))
# Get the center pixel (along axes=(1, 2))
features = features[:, features.shape[1]//2, features.shape[2]//2, :]
# Split into train and test data
x_train, x_test, y_train, y_test = train_test_split(features, targets,
                                                    test_size=0.2,
                                                    random_state=SEED)

# # Remove outliers
# std = np.std(y_train)
# mean = np.mean(y_train)
# cut_off = 3 * std
# mask = ma.masked_where(abs(y_train-mean) > cut_off, y_train)
# x_train = x_train[~mask.mask]
# y_train = y_train[~mask.mask]

# Shift values to remove negatives
x_train = np.apply_along_axis(add_min, 0, x_train)
x_test = np.apply_along_axis(add_min, 0, x_test)

# Log transform
x_train = safe_log(x_train)
x_test = safe_log(x_test)
y_train = safe_log(y_train)
y_test = safe_log(y_test)

# Normalize X
scaler_x = MinMaxScaler()
scaler_x.fit(x_train)
x_train = scaler_x.transform(x_train)
x_test = scaler_x.transform(x_test)

# Normalize y
scaler_y = MinMaxScaler()
scaler_y.fit(y_train.reshape(-1, 1))
y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# Identify features with 0 variance
zero_var_idx = np.where(np.var(x_train, axis=0) == 0)[0]
# Remove features with 0 variance
x_train = np.delete(x_train, zero_var_idx, -1)
x_test = np.delete(x_test, zero_var_idx, -1)

# # Identify features with high correlation
# high_corr_idx = get_corr_feats(x_train, min_corr=0.9)
# # Remove features with high correlation
# x_train = np.delete(x_train, high_corr_idx, -1)
# x_test = np.delete(x_test, high_corr_idx, -1)

# Convert data to float32
x_train = x_train.astype(np.float32)
y_train = y_train.astype(np.float32)
x_test = x_test.astype(np.float32)
y_test = y_test.astype(np.float32)


# ------------------- Feature selection ------------------------------------- #


# # Define progress monitoring object
# class tqdm_skopt(object):
#     """Progress bar object for functions with callbacks."""

#     def __init__(self, **kwargs):
#         self._bar = tqdm(**kwargs)

#     def __call__(self, res):
#         """Update bar with intermediate results."""
#         self._bar.update()


# # Create the RFE object and compute a cross-validated score.
# rf_fs = RandomForestRegressor(n_estimators=500, max_features=15,
#                               n_jobs=-1, random_state=SEED)
# cv_fs = KFold(n_splits=5, shuffle=True, random_state=SEED)
# rfecv = RFECV(estimator=rf_fs, step=1, cv=cv_fs, min_features_to_select=30,
#               scoring='neg_mean_squared_error', verbose=1)
# rfecv.fit(x_train, y_train)

# length = x_train.shape[-1]
# plt.figure()
# plt.xlabel('Number of features selected')
# plt.ylabel('Cross validation score (Negative MSE)')
# plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
# plt.savefig('rfe2.svg', format='svg')
# plt.show()

# rfe_mask = rfecv.support_
# rfe_rank = rfecv.ranking_
# rfe_scores = rfecv.grid_scores_

# np.save('rfe_scores2.npy', rfe_scores)
# np.save('rfe_rank2.npy', rfe_rank)
# np.save('rfe_mask2.npy', rfe_mask)

# rfe_mask = np.load('rfe_mask2.npy')

# x_train_fs = x_train[:, rfe_mask]
# x_test_fs = x_test[:, rfe_mask]


# ------------------- RF Hyperparameter Optimization ------------------------ #


# # Define estimator
# estimator = RandomForestRegressor(n_estimators=500, n_jobs=-1,
#                                   random_state=SEED)

# # Define cross-validation
# cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

# # Define search space
# n_features = x_train.shape[-1]

# space = []
# space.append(Integer(1, n_features, name='max_features'))
# space.append(Integer(10, 200, name='max_depth'))
# space.append(Integer(2, 100, name='min_samples_split'))
# space.append(Integer(1, 200, name='min_samples_leaf'))


# @use_named_args(space)
# def objective(**params):
#     """Return objective function score for estimator."""
#     # Set hyperparameters from space decorator
#     estimator.set_params(**params)

#     return -np.mean(cross_val_score(estimator, x_train, y_train, cv=cv,
#                                     n_jobs=-1,
#                                     scoring="neg_mean_squared_error"))


# n_calls = 200
# res_gp = gp_minimize(objective, space, n_calls=n_calls,
#                      random_state=SEED,
#                      callback=[tqdm_skopt(total=n_calls,
#                                           desc='Gaussian Process')])


# print(f'''Best parameters:
# - max_features={res_gp.x[0]}
# - max_depth={res_gp.x[1]}
# - min_samples_split={res_gp.x[2]}
# - min_samples_leaf={res_gp.x[3]}''')

# # Plot gp_minimize output
# plot_convergence(res_gp)
# plt.savefig("GP_convergence_3.svg")
# plot_objective(res_gp)
# plt.savefig("GP_objective_3.svg")
# plot_evaluations(res_gp)
# plt.savefig("GP_revaluations_3.svg")

# # Save optimizer
# dump(res_gp, "optimizers/gp_3.pkl")


# ------------------- Training ---------------------------------------------- #


rf = RandomForestRegressor(n_estimators=2500, n_jobs=-1,
                           random_state=SEED, criterion='mse', verbose=2)
rf.fit(x_train, y_train)


# ------------------- Testing ----------------------------------------------- #

# Predict on test set and reshape pred and true
y_pred = np.exp(scaler_y.inverse_transform(rf.predict(x_test).reshape(-1, 1)))
y_true = np.exp(scaler_y.inverse_transform(y_test.reshape(-1, 1)))

# Calculate metrics
r2 = np.exp(scaler_y.inverse_transform(rf.score(x_test, y_test)
                                       .reshape(1, -1)))[0][0]
mse = mean_squared_error(y_true, y_pred)
me = mean_error(y_true, y_pred)
mec = model_efficiency_coefficient(y_true, y_pred)
ccc = lin_ccc(y_true, y_pred)


# ------------------- Plotting ---------------------------------------------- #


fig, ax = plt.subplots(figsize=(8, 8))
ax.set_title(f'MSE: {mse:.3f}   ME: {me:.3f}   MEC: {mec:.3f}\
   CCC: {ccc:.3f}')
ax.scatter(y_true, y_pred, c=colors[0])
ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
        '--', lw=2, label='1:1 line', c=colors[1])
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
# Regression line
y_true1, y_pred1 = y_true.reshape(-1, 1), y_pred.reshape(-1, 1)
ax.plot(y_true1, LinearRegression().fit(y_true1, y_pred1).predict(y_true1),
        c=colors[2], lw=2, label='Trend')
ax.legend(loc='upper left')
ax.text(-11, 370,
        f'MSE: {mse:.3f}\nME:  {me:.3f}\nMEC: {mec:.3f}\nCCC: {ccc:.3f}',
        va='top', ha='left', linespacing=1.5, snap=True,
        bbox={'facecolor': 'white', 'alpha': 0, 'pad': 5})
plt.savefig('RF_mse_summary.svg')
plt.show()
