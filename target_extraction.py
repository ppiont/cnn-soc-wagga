#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 16:12:56 2020

@author: peter
"""

import pandas as pd
import os
import geopandas as gpd
import cartopy.io.shapereader as shpreader
from geopandas.tools import sjoin

if "data" not in os.getcwd():
    os.chdir("data/")

# read observation data
df = pd.read_csv("LUCAS_TOPSOIL_v1.csv")

# create geodataframe with geometry
gdf = gpd.GeoDataFrame(df, geometry = gpd.points_from_xy(df.GPS_LONG, df.GPS_LAT), crs = "EPSG:4326")

# get world boundaries
shpfilename = shpreader.natural_earth(resolution = '10m', category = 'cultural', name = 'admin_0_countries')

# read the shapefile using geopandas
world = gpd.read_file(shpfilename)

# read the german borders
germany = world[world['ADMIN'] == 'Germany']
germany.crs = "epsg:4326"

# select only observations within germany
targets = sjoin(gdf, germany, op = "within")

# remove unnecessary columns and save
targets = targets[df.columns.to_list()]
# POINT_ID is stored as a string in the Excel file, so I recast to integer
targets["POINT_ID"] = pd.to_numeric(targets["POINT_ID"], downcast='integer')
# I rename OC to SOC
targets = targets.rename(columns={"OC": "SOC"})
# I save as csv and pkl
targets.to_csv("germany_targets.csv")
targets.to_pickle("germany_targets.pkl")
