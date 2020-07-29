#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 14:41:37 2020

@author: peter
"""






import rasterio as rio
import os

bands = rio.open("data/germany_covars/CLM_CHE_BIO02.tif")


band = bands.read(1)
band.sum()

import geopandas as gpd
import pandas as pd
import os
import rasterio as rio
import rasterio.plot
import rasterio.features
import rasterio.warp
import matplotlib.pyplot as plt
import numpy as np
import glob
import re

if os.getcwd().split(r"/")[-1] != "data":
    os.chdir("data/")

# open raster to get crs
crs_raster = rio.open("germany_covars/CLM_CHE_BIO02.tif")


# Load targets
df = pd.read_csv("germany_targets.csv", index_col = 0)
gdf = gpd.GeoDataFrame(df, geometry = gpd.points_from_xy(df.GPS_LONG, df.GPS_LAT), crs = "EPSG:4326")
# Set crs to same as the raster's
gdf = gdf.to_crs(crs_raster.crs.data)

#Create list of filenames for cov rasters
file_list = glob.glob('germany_covars/*.tif')
#Create list of coordinates for each target
coordinates = ((x,y) for x, y in zip(gdf.GPS_LONG, gdf.GPS_LAT))

# for each coord(target), get surrounding from each covar raster, then merge to 415 band stack and output with 


def stack_target_bands(file_list, target_coords, outfile = r"stack_{}.tif", n_win_size = 3):
    #For each target coord
    for i, (lon, lat) in enumerate(target_coords, start = 1):
        
        #Get pixel coordinates from the first raster only
        with rio.open(file_list[0], 'r') as src0:
            py, px = src0.index(lon, lat)
            print(py, px)
            meta = src0.meta
            #Update meta to reflect the number of layers
            meta.update(count = len(file_list), dtype = rio.uint16)
            print(meta)
            
        #Create window
        window = rio.windows.Window(px - n_win_size//2, py - n_win_size//2, n_win_size, n_win_size)
        print(window)
        
        #Clip window around target for each layer and write bands to stack
        for id, file in enumerate(file_list, start = 1):
            with rio.open(file, 'r') as src1:
                #Read the data in the window
                #clip is a nbands * N * N numpy array
                clip = src1.read(window = window)
                #Add transform to metadata
                meta = src1.meta
                meta['width'], meta['height'] = n_win_size, n_win_size
                meta['transform'] = rio.windows.transform(window, src1.transform)
                meta.update(count = len(file_list), dtype = rio.uint16)
                
                print(np.shape(clip)) # something is wrong wit´h the clip shape!

                #Search filename without path and extension
                band_name = re.search('^(.*)\/(.*)(\..*)$', file).group(2)
                with rio.open(outfile.format(i), 'w', **meta) as dst:
                    dst.write_band(id, clip.astype(rio.uint16))
                    dst.set_band_description(id, band_name)
                    print(f"file {id} done")
    
                # with rio.open(outfile.format(i, n_win_size), 'w', **meta) as out:
                    #     out.write(clip)

stack_target_bands(file_list, coordinates)


# import re
# m = re.search('^(.*)\/(.*)(\..*)$', test)
# m.group(2)

# file_list = glob.glob('germany_covars/*.tif')
# # Count dtypes in rasters
# counts = list()
# for file in file_list:
#     with rio.open(file) as raster:
#         counts.append(raster.dtypes)

# counts_set = set(counts)
# counts_set

# for i in counts_set:
#     print(i, counts.count(i))




# STACK LAYERS
##############################################################################
import glob
file_list = glob.glob('germany_covars/*.tif')

# Read metadata of first file
with rio.open(file_list[0]) as src0:
    meta = src0.meta

# Update meta to reflect the number of layers
meta.update(count = len(file_list), dtype = rio.uint16)

# Read each layer and write it to stack
with rio.open('stack.tif', 'w', **meta) as dst:
    for id, layer in enumerate(file_list, start=1):
        with rio.open(layer) as src1:
            dst.write_band(id, src1.read(1).astype(rio.uint16))
        print(f"file {id} done")
            
##############################################################################


# EXTRACT WINDOWS AROUND TARGETS
###############################################################################
# coord = (gdf.geometry[i].x, gdf.geometry[i].y)

# infile = r"germany_covars/CLM_CHE_BIO02.tif"
# outfile = r'covar_{}.tif'
# coordinates = ((x,y) for x, y in zip(gdf.GPS_LONG, gdf.GPS_LAT))

# # NxN window
# N = 3

# # Open the raster
# with rio.open(infile) as dataset:

#     # Loop through list of coords
#     for i, (lon, lat) in enumerate(coordinates):

#         # Get pixel coordinates from map coordinates
#         py, px = dataset.index(lon, lat)
#         #print(f'Pixel Y, X coords: {py}, {px}')

#         # Build an NxN window
#         window = rio.windows.Window(px - N//2, py - N//2, N, N)
#         print(window)

#         # Read the data in the window
#         # clip is a nbands * N * N numpy array
#         clip = dataset.read(window=window)

#         # Write out a new file
#         meta = dataset.meta
#         meta['width'], meta['height'] = N, N
#         meta['transform'] = rio.windows.transform(window, dataset.transform)

#         with rio.open(outfile.format(i), 'w', **meta) as dst:
#             dst.write(clip)
##############################################################################


