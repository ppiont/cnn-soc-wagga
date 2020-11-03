#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 15:31:24 2020

@author: peter
"""

import rasterio as rio
from rasterio.plot import show
import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import os
import path
import re
import glob

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

with featdir:
    testcase = rio.open(r_list[500]).read(1)
    testcaseb = rio.open(r_list[500])

plt.imshow(testcase, cmap = "cool")
show(testcase)

testcaseb.bounds.to_list()

df2 = pd.DataFrame()
df2["test"] = testcaseb.bounds
# # Set geodataframe crs to raster crs
# with rio.open("features/stack_1.tif") as rref:
#     gdf = gdf.to_crs(rref.crs)

# # Load raster stack as array
# rref = rio.open("features/stack_1.tif")
# test_arr = test.read()
# test_arr.shape
# test_arr.geometry()

# targets.head()


