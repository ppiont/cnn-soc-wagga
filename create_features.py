#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 14:41:37 2020

@author: peter
"""

import geopandas as gpd
import pandas as pd
import os
import rasterio
import matplotlib.pyplot as plt
import numpy as np

os.chdir("data/")
!ls


raster = rasterio.open("germany_covars/CLM_CHE_BIO02.tif")

plt.imshow(raster.read(1))

raster.shape


raster_extent = np.asarray(raster.bounds)[[0,2,1,3]]

raster_extent

plt.imshow(raster.read(1), cmap='hot', extent=raster_extent)



# convert band to numpy array
nparray_test = np.asarray(raster.read(1))


# load targets
df = pd.read_csv("germany_targets.csv", index_col = 0).reset_index()
gdf = gpd.GeoDataFrame(df, geometry = gpd.points_from_xy(df.GPS_LONG, df.GPS_LAT), crs = "EPSG:4326")
