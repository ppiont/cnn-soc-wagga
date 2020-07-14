#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 14:41:37 2020

@author: peter
"""

import geopandas as gpd
import os
import rasterio
import rasterio.plot
import rasterio.features
import rasterio.warp
import matplotlib.pyplot as plt
import numpy as np

os.chdir("data/")

# open raster
raster = rasterio.open("germany_covars/CLM_CHE_BIO02.tif")

# plot raster
plt.imshow(raster.read(1))

# recast extent array to match matplotlib's
raster_extent = np.asarray(raster.bounds)[[0,2,1,3]]

# plot raster properly with coord axes
plt.imshow(raster.read(1), cmap='hot', extent=raster_extent)

# convert band to numpy array
raster_array = np.asarray(raster.read(1))





# load targets
df = pd.read_csv("germany_targets.csv", index_col = 0).reset_index()
gdf = gpd.GeoDataFrame(df, geometry = gpd.points_from_xy(df.GPS_LONG, df.GPS_LAT), crs = "EPSG:4326")

# gdf.plot()

# gdf.crs
# raster.crs
gdf = gdf.to_crs(raster.crs.data)



coord = (gdf.geometry[i].x, gdf.geometry[i].y)
    

import rasterio as rio

infile = r"germany_covars/CLM_CHE_BIO02.tif"
outfile = r'test_{}.tif'
coordinates = [(x,y) for x, y in zip(gdf.GPS_LONG, gdf.GPS_LAT)]

# NxN window
N = 3

# Open the raster
with rio.open(infile) as dataset:

    # Loop through list of coords
    for i, (lon, lat) in enumerate(len(coordinates)):

        # Get pixel coordinates from map coordinates
        py, px = dataset.index(lon, lat)
        print('Pixel Y, X coords: {}, {}'.format(py, px))

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

