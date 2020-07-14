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
import rasterio.plot
import rasterio.features
import rasterio.warp
import matplotlib.pyplot as plt
import numpy as np

os.chdir("data/")
!ls


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






# ENVELOPE BUFFER
# gdf2 = gdf.copy()
# gdf2.geometry = gdf.geometry.buffer(5000).envelope
# gdf2.plot()


# fig, ax = plt.subplots(figsize=(15, 15))
# rasterio.plot.show(raster.read(1), extent=raster_extent, ax = ax)
# gdf.plot(ax = ax, marker = ',', markersize = 3)




# world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
# germany = world[world.name == "Germany"]
# germany = germany.to_crs(raster.crs.data)
# ax = germany.plot(
#     color='white', edgecolor='black')
# ax.grid(linestyle="--", lw="0.5", zorder=1)
# ax.set_axisbelow(True)
# ax.set_title("Spatial distribution of soil samples") ## FIX FONTS
# plt.axis('scaled')

# # Plot
# gdf.plot(ax=ax, marker = '.', markersize = 1)

# # plt.savefig("figures/spatial_distribution_of_soil_samples.pdf")
# plt.show()