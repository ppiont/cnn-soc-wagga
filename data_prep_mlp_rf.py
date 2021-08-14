# -*- coding: utf-8 -*-
#%%
# Standard library imports
import pathlib

# Imports
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.model_selection import train_test_split

# Custom imports
from feat_eng.funcs import add_min, safe_log  # , highlight_corr_idx

# ------------------- Organization ------------------------------------------ #
DATA_DIR = pathlib.Path("data/")
SEED = 43
# ------------------- Read and prep data ------------------------------------ #

feature_names = pd.read_csv(
    DATA_DIR.joinpath("original_data/COVS250m_sel.csv"), header=None, names=["feature_name"], usecols=[1]
)

# Load target data
target_data = gpd.read_file(DATA_DIR.joinpath("targets/germany_targets.geojson"), driver="GeoJSON")
# Get target array
targets = target_data[["OC", "GPS_LAT", "GPS_LONG"]].to_numpy()
# Load feature array
features = np.load(DATA_DIR.joinpath("raw_features.npy"))
# Get the center pixel (along axes=(1, 2)) (for RF/MLP)
features = features[:, features.shape[1] // 2, features.shape[2] // 2, :]
# select subset of features
features = features[
    :,
    [
        59,
        84,
        85,
        110,
        111,
        112,
        113,
        114,
        115,
        116,
        117,
        118,
        123,
        124,
        125,
        126,
        155,
        156,
        157,
        330,
        331,
        332,
        333,
        334,
        335,
        336,
        340,
        342,
        353,
        355,
        357,
        359,
        387,
        389,
        390,
        392,
        394,
        396,
        398,
        400,
        402,
        403,
        404,
        405,
        406,
    ],
]

features = features.astype(np.float32)
# lowest val as nan
features[features == -32768] = np.nan
# get col means for imputation
col_mean = np.nanmedian(features, axis=(0), keepdims=True)
# get nan indices
inds = np.where(np.isnan(features))
# replace nan with col mean
features[inds] = np.take(col_mean, inds[1])

# Split into train and test data
x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=2 / 10, random_state=SEED)
#%%

# # Remove outliers
# std = np.std(y_train)
# mean = np.mean(y_train)
# cut_off = 3 * std
# mask = ma.masked_where(abs(y_train-mean) > cut_off, y_train)
# x_train = x_train[~mask.mask]
# y_train = y_train[~mask.mask]


# Shift values to remove negatives (CNN)
# am = np.min(x_train, axis=(0,1,2), keepdims=True)

# shift = np.abs(am)
# shift[am>=0] = 0
# shift
# x_train += shift


# # Shift values to remove negatives
# x_train = np.apply_along_axis(add_min, 0, x_train)
# x_test = np.apply_along_axis(add_min, 0, x_test)


# # Log transform
# x_train = safe_log(x_train)
# x_test = safe_log(x_test)
# y_train[:, 0] = safe_log(y_train[:, 0])
# y_test[:, 0] = safe_log(y_test[:, 0])


#%%

# Identify features with 0 variance
zero_var_idx = np.where(np.var(x_train, axis=(0)) == 0)[0]
# Remove features with 0 variance
x_train = np.delete(x_train, zero_var_idx, -1)
x_test = np.delete(x_test, zero_var_idx, -1)

# Also remove feature_names with that idx
feature_names = feature_names.drop(zero_var_idx).reset_index(drop=True)

# Convert data to float32
x_train = x_train.astype(np.float32)
y_train = y_train.astype(np.float32)
x_test = x_test.astype(np.float32)
y_test = y_test.astype(np.float32)

#%%

# Concat (y, x)
train = np.hstack((y_train, x_train))
test = np.hstack((y_test, x_test))

# # Save data
np.save(DATA_DIR.joinpath("train_45.npy"), train)
np.save(DATA_DIR.joinpath("test_45.npy"), test)

feature_names.to_csv(DATA_DIR.joinpath("feature_names_mlp_rf.csv"))

# %%
