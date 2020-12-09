#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 17:30:56 2020.

@author: peter
"""

import pandas as pd
import numpy as np
import pathlib
from feat_eng.target_extract import target_extract
from feat_eng.feature_extract import feature_extract


# Define data directory
data_dir = pathlib.Path('data/')

# ------------------- Targets ----------------------------------------------- #

# Extract targets
# targets = target_extract('data/LUCAS_TOPSOIL_v1.csv', 'Germany', 'GPS_LAT',
#                       'GPS_LONG')

# Save targets
# targets.to_file(data_dir.joinpath('targets.geojson'),
#                 driver='GeoJSON')

# ------------------- Features ---------------------------------------------- #

# Get target and feature paths
# t_path = data_dir.joinpath('germany_targets.geojson'
# r_path = data_dir.joinpath('germany_covars/'

# Extract features
# feats = feature_extract(t_path, r_path, 15)

# Save features
# np.save("raw_features.npy", feats)

# ------------------- Split numerical and categorical ------------------------#

# Read covariate table
covs_meta = pd.read_csv(data_dir.joinpath('COVS250m_sel.csv'))

# Create mask from the column that specifies categorical
mask = covs_meta.categorical.values

# Make mask numpy conforming
mask = np.where(mask == 'yes', True, False)

# Load feats
feats = np.load(data_dir.joinpath('raw_features.npy'))

# Apply mask to extract numerical and categorical and save both
numerical = feats[:, :, :, ~mask]
np.save(data_dir.joinpath('numerical_feats.npy'), numerical)
del numerical  # delete to free memory

categorical = feats[:, :, :, mask]
np.save(data_dir.joinpath('categorical_feats.npy'), categorical)
del categorical  # delete to free memory

del feats  # delete to free memory
