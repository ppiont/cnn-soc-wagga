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
import seaborn as sns

mpl.style.use('tableau-colorblind10')
mpl.rcParams.update({"lines.linewidth": 1, "font.family": "serif", 
                     "xtick.labelsize": "small", "ytick.labelsize": "small", 
                     "xtick.major.size" : 0, "xtick.minor.size" : 0, 
                     "ytick.major.size" : 0, "ytick.minor.size" : 0, 
                     "axes.titlesize":"medium", "figure.titlesize": "medium", 
                     "figure.figsize": (5,5), "figure.dpi": 300, 
                     "figure.autolayout": True, "savefig.format": "pdf", 
                     "savefig.transparent": True, "image.cmap": "viridis"})

if os.getcwd().split(r"/")[-1] != "data":
    os.chdir("data/")
    
fig_path = "/home/peter/cnn-soc-wagga/figures/"

df = pd.read_csv("germany_targets.csv", index_col = 0)
df['OC_z'] = (df.OC - df.OC.mean())/df.OC.std(ddof=0)
gdf = gpd.GeoDataFrame(df, geometry = gpd.points_from_xy(df.GPS_LONG, df.GPS_LAT), crs = "EPSG:4326")
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))



################################ CARTOPY TEST ################################
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from mpl_toolkits.axes_grid1 import make_axes_locatable
# from matplotlib.offsetbox import AnchoredText


# def make_colorbar(ax, mappable, **kwargs):
#     from mpl_toolkits.axes_grid1 import make_axes_locatable
#     import matplotlib as mpl

#     divider = make_axes_locatable(ax)
#     orientation = kwargs.pop('orientation', 'vertical')
#     if orientation == 'vertical':
#         loc = 'right'
#     elif orientation == 'horizontal':
#         loc = 'bottom'
        
#     cax = divider.append_axes(loc, '5%', pad='3%', axes_class=mpl.pyplot.Axes)
#     ax.get_figure().colorbar(mappable, cax=cax, orientation=orientation)


# get country borders
resolution = '10m'
category = 'cultural'
name = 'admin_0_countries'

shpfilename = shpreader.natural_earth(resolution, category, name)

# read the shapefile using geopandas
df = gpd.read_file(shpfilename)

# read the german borders
poly = df.loc[df['ADMIN'] == 'Germany']['geometry'].values[0]

# create ax
fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=ccrs.EuroPP()))
# set title
plt.title('Deutschland')

# add geometries and features
ax.coastlines(resolution='10m', alpha = 0.5)
ax.add_feature(cfeature.BORDERS, alpha = 0.5)
ax.add_geometries(poly, crs=ccrs.PlateCarree(), facecolor='none', edgecolor='0')

# convert gpd to same proj as cartopy map
crs_proj4 = ccrs.EuroPP().proj4_init
gdf_utm32 = gdf.to_crs(crs_proj4)

# Plot
gdf_utm32.plot(ax = ax, marker = '.', markersize = 10, column = "OC")

# mappable = gdf_utm32.plot(ax = ax, marker = '.', markersize = 10, column = "OC")

ax.set_extent([5.5, 15.5, 46.5, 55.5], crs=ccrs.PlateCarree())

# make_colorbar(ax, mappable, orientation='vertical')

# plt.savefig(os.path.join(fig_path, "testplot.pdf"), bbox_inches = 'tight', pad_inches = 0)
plt.show()







############################### SOC LEVEL PLOT ###############################


# Restrict to Germany
ax = world[world.name == "Germany"].plot(
    color='white', edgecolor='black')
ax.set_title("Soil sample distribution", loc = "left")
ax.grid(linestyle="--", lw="0.5", zorder=1)
ax.set_axisbelow(True)
plt.axis('scaled')
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

# colors = plt.cm.viridis(gdf.OC.ravel() / float(max(gdf.OC.ravel())))
# sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=1))
# # fake up the array of the scalar mappable. Urgh…
# sm._A = []
# plt.colorbar(sm)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad="3%")
cax.set_title("SOC (g/kg)", loc = "right")
# gdf.crs = "EPSG:4839"

# Plot
gdf.plot(ax = ax, marker = '.', markersize = 10, column = "OC", legend=True, cax = cax)
# plt.savefig(os.path.join(fig_path, "germanyplot.pdf"), bbox_inches = 'tight', pad_inches = 0)
plt.show()




################################# POINT DISTRO#################################

# Restrict to Germany
ax = world[world.name == "Germany"].plot(
    color='white', edgecolor='black')
ax.grid(linestyle="--", lw="0.5", zorder=1)
ax.set_axisbelow(True)
ax.set_title("Spatial distribution of soil samples") ## FIX FONTS
plt.axis('scaled')
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
gdf.crs = "EPSG:4839"

# Plot
gdf.plot(ax = ax, marker = '.', markersize = 10)

# plt.savefig(os.path.join(fig_path, "spatial_distribution_of_soil_samples.pdf"), bbox_inches = 'tight', pad_inches = 0)
plt.show()


#################################### HIST ####################################

sns.set_context("paper")
sns.set_style("whitegrid")

x = df.OC

ax = sns.distplot(x)
ax.set_title("Distribution plot of sampled SOC")
ax.set_xlabel("SOC (g/kg)")
sns.despine()
plt.savefig(os.path.join(fig_path, "soc_distplot.pdf"), bbox_inches = 'tight', pad_inches = 0)

ax = sns.boxplot(x)
ax.set_title("Boxplot of sampled SOC")
ax.set_xlabel("SOC (g/kg)")
sns.despine()
plt.savefig(os.path.join(fig_path, "soc_boxplot.pdf"), bbox_inches = 'tight', pad_inches = 0)
















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

