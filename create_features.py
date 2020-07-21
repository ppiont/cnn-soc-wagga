#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 14:41:37 2020

@author: peter
"""

import geopandas as gpd
import pandas as pd
import os
import rasterio as rio
import rasterio.plot
import rasterio.features
import rasterio.warp
import matplotlib.pyplot as plt
import numpy as np

os.chdir("data/")

# open raster to get crs
raster = rasterio.open("germany_covars/CLM_CHE_BIO02.tif")

# Load targets
df = pd.read_csv("germany_targets.csv", index_col = 0).reset_index()
gdf = gpd.GeoDataFrame(df, geometry = gpd.points_from_xy(df.GPS_LONG, df.GPS_LAT), crs = "EPSG:4326")
gdf = gdf.to_crs(raster.crs.data)



# STACK LAYERS
##############################################################################
import glob
file_list = glob.glob('germany_covars/*.tif')

# Read metadata of first file
with rio.open(file_list[0]) as src0:
    meta = src0.meta

# Update meta to reflect the number of layers
meta.update(count = len(file_list), dtype = "int16")

# Read each layer and write it to stack
with rio.open('stack.tif', 'w', **meta) as dst:
    for id, layer in enumerate(file_list, start=1):
        with rio.open(layer) as src1:
            dst.write_band(id, src1.read(1).astype("int16"))
        print(f"file {id} done")
            
##############################################################################


# EXTRACT WINDOWS AROUND TARGETS
###############################################################################
coord = (gdf.geometry[i].x, gdf.geometry[i].y)

infile = r"germany_covars/CLM_CHE_BIO02.tif"
outfile = r'covar_{}.tif'
coordinates = ((x,y) for x, y in zip(gdf.GPS_LONG, gdf.GPS_LAT))

# NxN window
N = 3

# Open the raster
with rio.open(infile) as dataset:

    # Loop through list of coords
    for i, (lon, lat) in enumerate(coordinates):

        # Get pixel coordinates from map coordinates
        py, px = dataset.index(lon, lat)
        #print(f'Pixel Y, X coords: {py}, {px}')

        # Build an NxN window
        window = rio.windows.Window(px - N//2, py - N//2, N, N)
        print(window)

        # Read the data in the window
        # clip is a nbands * N * N numpy array
        clip = dataset.read(window=window)

        # Write out a new file
        meta = dataset.meta
        meta['width'], meta['height'] = N, N
        meta['transform'] = rio.windows.transform(window, dataset.transform)

        with rio.open(outfile.format(i), 'w', **meta) as dst:
            dst.write(clip)
##############################################################################

## Count dtypes in rasters
# counts = list()
# for file in file_list:
#     with rio.open(file) as raster:
#         counts.append(raster.dtypes)

# counts_set = set(counts)
# counts_set

# for i in counts_set:
#     print(i, counts.count(i))


# for each coord(target), get surrounding from each covar raster, then merge to 415 band stack and output with 

import glob
covar_tifs = glob.glob('germany_covars/*.tif')

def stack_target_bands():
    for (lon, lat) in coordinates:
        #get pixel cooridnates from the first raster only
        with rio.open() as dst
        py, px = dst.index(lon, lat)
        for raster in covar_tifs:
            
        