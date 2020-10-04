#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 23:31:08 2020

@author: peter
"""

import os
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

plt.style.use('tableau-colorblind10')
mpl.rcParams.update({"lines.linewidth": 1, "font.family": "serif", 
                     "xtick.labelsize": "small", "ytick.labelsize": "small", 
                     "xtick.major.size" : 0, "xtick.minor.size" : 0, 
                     "ytick.major.size" : 0, "ytick.minor.size" : 0, 
                     "axes.titlesize":"medium", "figure.titlesize": "medium", 
                     "figure.figsize": (5, 5), "figure.dpi": 600, 
                     "figure.autolayout": True, "savefig.format": "pdf", 
                     "savefig.transparent": True, "image.cmap": "magma_r"})

if os.getcwd().split(r"/")[-1] != "data":
    os.chdir("data/")
    
fig_path = "../figures/"

df = pd.read_csv("germany_targets.csv", index_col = 0)
gdf = gpd.GeoDataFrame(df, geometry = gpd.points_from_xy(df.GPS_LONG, df.GPS_LAT), crs = "EPSG:4326")

# get country borders
shpfilename = shpreader.natural_earth(resolution = "10m", category = "cultural", name = "admin_0_countries")

# read the shp
shape = gpd.read_file(shpfilename)

# extract germany geom
poly = shape.loc[shape['ADMIN'] == 'Germany']['geometry'].values[0]

# create fig, ax
fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=ccrs.EuroPP()))

# add geometries and features
ax.coastlines(resolution="10m", alpha = 0.3)
ax.add_feature(cfeature.BORDERS, alpha = 0.3)
ax.add_geometries(poly, crs=ccrs.PlateCarree(), facecolor = "none", edgecolor = '0.5')

# convert gpd to same proj as cartopy map
crs_proj4 = ccrs.EuroPP().proj4_init
gdf_utm32 = gdf.to_crs(crs_proj4)

# Plot
gdf_utm32.plot(ax = ax, marker = ".", markersize = 10, column = "OC", legend = True)

# set extent of map
ax.set_extent([5.5, 15.5, 46.5, 55.5], crs=ccrs.PlateCarree())

# fix axes pos
map_ax = fig.axes[0]
leg_ax = fig.axes[1]
map_box = map_ax.get_position()
leg_box = leg_ax.get_position()
leg_ax.set_position([leg_box.x0, map_box.y0, leg_box.width, map_box.height])
map_ax.set_title("Sample distribution", pad = 10)
leg_ax.set_title("SOC (g/kg)", pad = 10)

# save and show fig
plt.savefig(os.path.join(fig_path, "sample_distribution_soc2.pdf"), bbox_inches = 'tight', pad_inches = 0)
plt.show()