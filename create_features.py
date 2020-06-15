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
nparray_test = np.asarray(raster.read(1))

# load targets
df = pd.read_csv("germany_targets.csv", index_col = 0).reset_index()
gdf = gpd.GeoDataFrame(df, geometry = gpd.points_from_xy(df.GPS_LONG, df.GPS_LAT), crs = "EPSG:4326")

# gdf.plot()

# gdf.crs
# raster.crs
gdf = gdf.to_crs(raster.crs.data)

gdf.head()








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