# -*- coding: utf-8 -*-

# Standard library imports
import pathlib

# Imports
import numpy as np
import numpy.ma as ma
import geopandas as gpd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# Custom imports
from feat_eng.funcs import add_min, safe_log  # , highlight_corr_idx

# ------------------- Organization ------------------------------------------ #


DATA_DIR = pathlib.Path('data/')
SEED = 43


# ------------------- Read and prep data ------------------------------------ #


# Load target data
target_data = gpd.read_file(DATA_DIR.joinpath('targets/germany_targets.geojson'),
                            driver='GeoJSON')
# Get target array
targets = target_data.OC.values
# Load feature array
features = np.load(DATA_DIR.joinpath('numerical_feats.npy'))
# Get the center pixel (along axes=(1, 2))
features = features[:, features.shape[1]//2, features.shape[2]//2, :]
# Split into train and test data
x_train, x_test, y_train, y_test = train_test_split(features, targets,
                                                    test_size=0.2,
                                                    random_state=SEED)

# Remove outliers
std = np.std(y_train)
mean = np.mean(y_train)
cut_off = 3 * std
mask = ma.masked_where(abs(y_train-mean) > cut_off, y_train)
x_train = x_train[~mask.mask]
y_train = y_train[~mask.mask]

# Shift values to remove negatives
x_train = np.apply_along_axis(add_min, 0, x_train)
x_test = np.apply_along_axis(add_min, 0, x_test)

# Log transform
x_train = safe_log(x_train)
x_test = safe_log(x_test)
y_train = safe_log(y_train)
y_test = safe_log(y_test)

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

# # Normalize X
# scaler_x = MinMaxScaler()
# scaler_x.fit(x_train)
# x_train = scaler_x.transform(x_train)
# x_test = scaler_x.transform(x_test)

# # Normalize y
# scaler_y = MinMaxScaler()
# scaler_y.fit(y_train.reshape(-1, 1))
# y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
# y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# Convert data to float32
x_train = x_train.astype(np.float32)
y_train = y_train.astype(np.float32)
x_test = x_test.astype(np.float32)
y_test = y_test.astype(np.float32)

# Concat (y, x)
train = np.hstack((np.expand_dims(y_train, axis=-1), x_train))
test = np.hstack((np.expand_dims(y_test, axis=-1), x_test))

# Save data
np.save(DATA_DIR.joinpath('train_log_no_outlier.npy'), train)
np.save(DATA_DIR.joinpath('test_log_no_outlier.npy'), test)
