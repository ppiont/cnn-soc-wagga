#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 16:12:56 2020

@author: peter
"""

import pandas as pd
pd.options.display.max_columns = 500
pd.options.display.max_rows = 500
pd.options.display.precision = 5

df = pd.read_csv("LUCAS_TOPSOIL_v1.csv")

import geopandas as gpd
import cartopy.io.shapereader as shpreader
from geopandas.tools import sjoin

gdf = gpd.GeoDataFrame(df, geometry = gpd.points_from_xy(df.GPS_LONG, df.GPS_LAT), crs = "EPSG:4326")

# get country borders
resolution = '10m'
category = 'cultural'
name = 'admin_0_countries'

shpfilename = shpreader.natural_earth(resolution = '10m', category = 'cultural', name = 'admin_0_countries')

# read the shapefile using geopandas
world = gpd.read_file(shpfilename)

world.geometry.crs


# read the german borders
germany = world[world['ADMIN'] == 'Germany']

germany.to_crs()

germany.crs



targets = sjoin(gdf, germany)


world2 = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
germany2 = world2[world2.name == 'Germany']

germany2.head()
