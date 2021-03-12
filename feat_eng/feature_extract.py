#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 14:41:37 2020.

@author: peter
"""

import geopandas as gpd
import rasterio as rio
import rasterio.features
import rasterio.warp
import pathlib
import numpy as np


def feature_extract(target_path, raster_dir, win_size):
    """Do something.

    Parameters
    ----------
    target_path : str
        The path to the targets file.
    raster_dir : str
        The path to the raster directory.

    Returns
    -------
    Returns something.

    """
    # Assign paths to pathlib.Path class
    target_path = pathlib.Path(target_path)
    raster_dir = pathlib.Path(raster_dir)

    # Create list of rasters to process
    r_list = list(r.name for r in raster_dir.glob('*.tif'))

    # Get reference CRS from the input raster
    with rio.open(raster_dir.joinpath(r_list[0]), 'r') as src:
        ref_crs = src.crs.to_wkt()

    # Read the targets and reproject them to the reference CRS
    targets = gpd.read_file(target_path)
    targets = targets.to_crs(ref_crs)

    # Create a list of coordinates for all targets
    coordinates = [(x, y) for x, y in zip(targets.geometry.x,
                                          targets.geometry.y)]

    # Iterate over coordinates and extract rasters clipped to window
    for i, (x, y) in enumerate(coordinates):

        # Get meta data and pixel coordinates from the first raster
        with rio.open(raster_dir.joinpath(r_list[0]), "r") as src:
            py, px = src.index(x, y)
            # meta = src.meta # not sure this is necessary

        # Create window
        window = rio.windows.Window(px - win_size//2, py - win_size//2,
                                    win_size, win_size)

        for id, band in enumerate(r_list):
            if id > 0:
                with rio.open(raster_dir.joinpath(band)) as src:
                    clip = np.expand_dims(src.read(1, window=window), -1)
                    inner_array = np.append(inner_array, clip, -1)
            else:
                with rio.open(raster_dir.joinpath(band)) as src:
                    clip = src.read(1, window=window)
                    inner_array = np.expand_dims(clip, -1)

        if i > 0:
            outer_array = np.append(outer_array, np.expand_dims(inner_array, 0), 0)

        else:
            outer_array = np.expand_dims(inner_array, 0)
        print(f"Target {i} done [E/N = {(x,y)}]")

    return outer_array


# t_path = '/home/peter/GDrive/Thesis/cnn-soc-wagga/data/targets/germany_targets.geojson'
# r_path = '/home/peter/GDrive/Thesis/cnn-soc-wagga/data/original_data/germany_covars'

# new_array = feature_extract(t_path, r_path, 15)

# np.save("testdata.npy", new_array)
