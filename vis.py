#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 23:31:08 2020

@author: peter
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

mpl.style.use('tableau-colorblind10')
mpl.rcParams.update({"lines.linewidth": 1, "font.family": "serif", "xtick.labelsize": "small", "ytick.labelsize": "small", "xtick.major.size" : 0, "xtick.minor.size" : 0, "ytick.major.size" : 0, "ytick.minor.size" : 0, 
                     "axes.titlesize":"medium", "figure.titlesize": "medium", "figure.figsize": (5,5), "figure.dpi": 300, "figure.autolayout": True, "savefig.format": "pdf", "savefig.transparent": True})

if os.getcwd().split(r"/")[-1] != "data":
    os.chdir("data/")
    
fig_path = "figures/"

df = pd.read_csv("germany_targets.csv", index_col = 0)
df['OC_z'] = (df.OC - df.OC.mean())/df.OC.std(ddof=0)
gdf = gpd.GeoDataFrame(df, geometry = gpd.points_from_xy(df.GPS_LONG, df.GPS_LAT), crs = "EPSG:4326")
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))


############################### SOC LEVEL PLOT ###############################

# Restrict to Germany
ax = world[world.name == "Germany"].plot(
    color='white', edgecolor='black')
ax.set_title("Spatial distribution of soil samples, Germany")
ax.grid(linestyle="--", lw="0.5", zorder=1)
ax.set_axisbelow(True)
plt.axis('scaled')
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cax.set_label("SOC (g/kg)")
gdf.crs = "EPSG:4839"

# Plot
gdf.plot(ax=ax, marker = '.', markersize = 10, column = "OC", legend=True, cax = cax)
plt.show()

# plt.savefig("figures/germanyplot.pdf")





################################# POINT DISTRO#################################

# Restrict to Germany
ax = world[world.name == "Germany"].plot(
    color='white', edgecolor='black')
ax.grid(linestyle="--", lw="0.5", zorder=1)
ax.set_axisbelow(True)
ax.set_title("Spatial distribution of soil samples") ## FIX FONTS
plt.axis('scaled')
gdf.crs = "EPSG:4839"

# Plot
gdf.plot(ax=ax, marker = '.', markersize = 1)

# plt.savefig("figures/spatial_distribution_of_soil_samples.pdf")
plt.show()



# # Load targets
# targets = gpd.read_file("targets.geojson")
# # Load raster
# raster = rio.open("germany_covars/CLM_CHE_BIO02.tif", "r")
# # Reproject targets as geojson auto projects to WGS84 upon save
# targets.crs = raster.crs

# # Plot targets on raster for testing projections
# fig, ax = plt.subplots(figsize = (10, 10), dpi = 300)
# rio.plot.show(raster, ax=ax)
# gdf.plot(ax=ax, marker = '.', markersize = 1, column = "OC", legend=True)





# # open raster
# raster = rasterio.open("germany_covars/CLM_CHE_BIO02.tif")

# # plot raster
# plt.imshow(raster.read(1))

# # recast extent array to match matplotlib's
# raster_extent = np.asarray(raster.bounds)[[0,2,1,3]]

# # plot raster properly with coord axes
# plt.imshow(raster.read(1), cmap='hot', extent=raster_extent)

# # convert band to numpy array
# raster_array = np.asarray(raster.read(1))

