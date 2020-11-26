#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 15:31:24 2020

@author: peter
"""

import os
import path
import re
import glob
import rasterio as rio
import numpy as np
import geopandas as gpd
import pandas as pd

# Set some pandas options
pd.options.display.max_columns = 500
pd.options.display.max_rows = 500
pd.options.display.precision = 5

# Go to data folder if not already in it
if os.getcwd().split(r"/")[-1] != "data":
    os.chdir("data/")

# Save feature dir as Path
d = path.Path(os.getcwd())
featdir = d / "features"

df = pd.read_pickle("germany_targets.pkl")
targets = (gpd.GeoDataFrame(df)
           [["SOC", "POINT_ID", "GPS_LAT", "GPS_LONG", "geometry"]]
           .reset_index(drop=True)
           )


# I define a func to assign multiple columns with some function
# based on some condition
def multi_assign(df, transform_fn, condition):
    df_to_use = df.copy()

    return (df_to_use
            .assign(
                **{col: transform_fn(df_to_use[col])
                   for col in condition(df_to_use)})
            )


# I define a function to downcast numeric values into the type allowing for
# the smallest memory use
def downcast_all(df, target_type, initial_type=None):

    if initial_type is None:
        initial_type = target_type

    df_to_use = df.copy()

    def transform_fn(x):

        return pd.to_numeric(x, downcast=target_type)

    def condition(x):

        return list(x
                    .select_dtypes(include=[initial_type])
                    .columns)

    return multi_assign(df_to_use, transform_fn, condition)


targets = (targets
           .pipe(downcast_all, "float")
           .pipe(downcast_all, "integer")
           )


# Get sorted list of raster files
# Jump into feature directory and grab file names of GTiffs
with featdir:
    r_list = glob.glob("*.tif")
# Define function to extract numbers from file names
def numbers(x):
    return(int(re.split("_|\.", x)[1]))
# Sort based on numerical pattern (instead of alphabetical)
r_list = sorted(r_list, key = numbers)

# Create list of raster bounds
bounds_list = []
with featdir:
    for file in r_list:
        with rio.open(file) as raster:
            bounds_list.append(np.array(raster.bounds, dtype = np.int32))
            print(file, "done")

# Create list of feature arrays
array_list = []
with featdir:
    for file in r_list:
        with rio.open(file) as raster:
            array_list.append(np.moveaxis(raster.read(), 0, 2))
            print(file, "done")

# Insert features and bounds
targets.insert(1, "features", array_list)
targets.insert(len(targets.columns)-1, "bounds", bounds_list)

# Rename OC col to SOC
targets.rename(columns = {"OC": "SOC"}, inplace = True)

# Save to json
targets[["POINT_ID", "GPS_LAT", "GPS_LONG", "geometry"]].to_file("IDs_geom.json", driver="GeoJSON")
targets[["SOC", "features", "POINT_ID", "GPS_LAT", "GPS_LONG", "bounds"]].to_pickle("targs_feats_IDs.pkl")






# # Test if rotations work
# test_list = []
# for i in range(1, 4):
#     temp = [np.rot90(array, k = i, axes = (1, 0)) for array in array_list]
#     test_list.append(temp)
    
# # Plot test rotations
# fig, axs = plt.subplots(2, 2)
# fig.suptitle('Rotations')
# axs[0, 0].imshow(array_list[0][:,:,42])
# axs[0, 0].set_title("0 deg")
# axs[0, 1].imshow(test_list[0][0][:,:,42])
# axs[0, 1].set_title("90 deg")
# axs[1, 0].imshow(test_list[1][0][:,:,42])
# axs[1, 0].set_title("180 deg")
# axs[1, 1].imshow(test_list[2][0][:,:,42])
# axs[1, 1].set_title("270 deg")
# plt.tight_layout()