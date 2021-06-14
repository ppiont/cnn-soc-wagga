# -*- coding: utf-8 -*-
#%%
# Standard library imports
import pathlib
# Imports
import numpy as np
import geopandas as gpd
# ------------------- Organization ------------------------------------------ #
DATA_DIR = pathlib.Path('data/')
SEED = 43
# ------------------- Read and prep data ------------------------------------ #
# Load target data
target_data = gpd.read_file(DATA_DIR.joinpath('targets/germany_targets.geojson'),
                            driver='GeoJSON')
# Get target array
targets = target_data[['OC', 'GPS_LAT', 'GPS_LONG']].to_numpy()
# Load feature array
features = np.load(DATA_DIR.joinpath('raw_features.npy'))
features = features[:, :, :, [59, 84, 85, 110, 111, 112, 113, 114, 115, 116, 117, 118,
                        123, 124, 125, 126, 155, 156, 157, 330, 331, 332, 333,
                        334, 335, 336, 340, 342, 353, 355, 357, 359, 387, 389,
                        390, 392, 394, 396, 398, 400, 402, 403, 404, 405, 406]]

# Set to float32 to make nans
features = features.astype(np.float32)
# lowest val as nan
features[features == -32768] = np.nan
# get col means for imputation
col_mean = np.nanmean(features, axis=(0, 1, 2), keepdims=True)
# get nan indices
inds = np.where(np.isnan(features))
# replace nan with col mean
features[inds] = np.take(col_mean, inds[3])

#%%
# Identify features with 0 variance
zero_var_idx = np.where(np.var(features, axis=(0, 1, 2)) == 0)[0]
# Remove features with 0 variance
features = np.delete(features, zero_var_idx, -1)
# dtype
targets = targets.astype(np.float32)
features = features.astype(np.float32)

#%%
# Save data
np.save('data/cnn_targets.npy', targets)
np.save('data/cnn_features.npy', features)