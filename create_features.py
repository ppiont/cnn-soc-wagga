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
import glob
import re


pd.options.display.max_columns = 500
pd.options.display.max_rows = 500
pd.options.display.precision = 5

if os.getcwd().split(r"/")[-1] != "data":
    os.chdir("data/")

if not os.path.exists("features"):
    os.makedirs("features")

# Load first feature raster to get crs
crs_raster = rio.open("germany_covars/CLM_CHE_BIO02.tif")

# Load targets
df = pd.read_csv("germany_targets.csv", index_col=0)
gdf = (gpd.GeoDataFrame(df,
                        geometry=gpd.points_from_xy(df.GPS_LONG, df.GPS_LAT),
                        crs="EPSG:4326"))
# Set crs to same as the raster's
gdf = gdf.to_crs(crs_raster.crs.data)

# Create list of filenames for cov rasters
file_list = sorted(glob.glob("germany_covars/*.tif"))

# Create list of coordinates for each target
coordinates = [(x, y) for x, y in zip(gdf.geometry.x, gdf.geometry.y)]


# For each coord(target), get surrounding from each covar raster,
# then merge to 415 band stack and output with
def stack_target_bands(file_list, target_coords, outfolder="features/",
                       outfile=r"stack_{}.tif", n_win_size=3):
    """Stack all bands and extract data around each target coord
    with a chosen window size"""
    # For each target coord
    for i, (x, y) in enumerate(target_coords, start=1):

        # Get pixel coordinates and meta from the first raster only
        with rio.open(file_list[0], "r") as src0:
            py, px = src0.index(x, y)
            meta = src0.meta

        # Create window
        window = rio.windows.Window(px - n_win_size//2, py - n_win_size//2,
                                    n_win_size, n_win_size)

        # Update window related meta
        meta['width'], meta['height'] = n_win_size, n_win_size
        meta['transform'] = rio.windows.transform(window, src0.transform)
        # Update band count in meta
        meta.update(count=len(file_list), dtype=rio.int32)
        with rio.open((outfolder + outfile.format(i)), "w", **meta) as dst:
            for id, band in enumerate(file_list, start=1):
                with rio.open(band) as src1:
                    band_name = re.search(r"^(.*)\/(.*)(\..*)$", band).group(2)
                    # clip = src1.read(1, window = window)
                    dst.write_band(id, src1.read(1, window=window)
                                   .astype(rio.int32))
                    dst.set_band_description(id, band_name)
        print(f"Target {i} done [E/N = {(x,y)}]")


stack_target_bands(file_list, coordinates, n_win_size=15)


# import re
# m = re.search('^(.*)\/(.*)(\..*)$', test)
# m.group(2)
