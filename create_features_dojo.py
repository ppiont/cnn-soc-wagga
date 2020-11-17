#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 15:31:24 2020

@author: peter
"""

import os
import sys
import path
import re
import glob
import rasterio as rio
from rasterio.plot import show
import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import rotate


pd.options.display.max_columns = 500
pd.options.display.max_rows = 500
pd.options.display.precision = 5

# Go to data folder if not already in it
if os.getcwd().split(r"/")[-1] != "data":
    os.chdir("data/")

# Save feature dir as Path
d = path.Path(os.getcwd())
featdir = d / "features"

# Load targets
gdf = gpd.read_file("germany_targets.csv", GEOM_POSSIBLE_NAMES = "geometry", 
            KEEP_GEOM_COLUMNS = "NO").set_crs("EPSG:4326")

# Remove unwanted cols and reindex
targets = gdf[["OC", "POINT_ID", "GPS_LAT", "GPS_LONG", "geometry"]].reset_index(drop = True)

## Get sorted list of raster files
# Jump into feature directory and grab file names of GTiffs
with featdir:
    r_list = glob.glob("*.tif")
# Define function to extract numbers from file names
def numbers(x):
    return(int(re.split("_|\.", x)[1]))
# Sort based on numerical pattern (instead of alphabetical)
r_list = sorted(r_list, key = numbers)

# with featdir:
#     with rio.open(r_list[0]) as raster:
#         print(raster.read().shape)

# Create list of raster bounds and add to df
bounds_list = []
with featdir:
    for file in r_list:
        with rio.open(file) as raster:
            bounds_list.append(list(raster.bounds))
            print(file, "done")

targets["bounds"] = bounds_list # doesn't work, it is pivoted on maxy etc, fix


# Create list of raster array and add to df
array_list = []
with featdir:
    for file in r_list:
        with rio.open(file) as raster:
            array_list.append(np.moveaxis(raster.read(), 0, 2))
            print(file, "done")

targets["features"] = array_list

# targets.to_file("targets.json", driver="GeoJSON")

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