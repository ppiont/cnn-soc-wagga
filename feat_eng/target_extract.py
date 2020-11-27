#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 16:12:56 2020.

@author: peter
"""

import pandas as pd
import geopandas as gpd
import cartopy.io.shapereader as shpreader


def target_extract(path, country, lat_col, lon_col, crs='EPSG:4326'):
    """
    Extract records whose location lies within a given country.

    Args:
        path (str): Path to the input table.
        country (str): Country to clip to.
        lat_col (str): Name of latitude column.
        lon_col (str): Name of longitude column.
        crs (str): CRS of the input geometry. Defaults to 'EPSG:4326'.

    Returns
    -------
        A CSV containing the input dataframe's values clipped to the country.

    """
    # Read input from path
    df = pd.read_table('data/LUCAS_TOPSOIL_v1.csv', sep=None, engine='python')

    # Create GeoDataFrame with geometry
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(
                                df[lon_col], df[lat_col]), crs=crs)

    # Get and read the country boundaries
    world = gpd.read_file(shpreader.natural_earth(resolution='10m',
                                                  category='cultural',
                                                  name='admin_0_countries')
                          )

    country_geom = world[world['ADMIN'] == country.capitalize()].geometry
    country_geom.crs = 'EPSG:4326'

    # Clip to records within country
    subset = gpd.clip(gdf, country_geom).reset_index(drop=True)
    # subset = gdf.cx[country_geom]

    return subset


# targets = target_extract('data/LUCAS_TOPSOIL_v1.csv', 'Germany', 'GPS_LAT',
#                       'GPS_LONG')

# targets.to_file("germany_targets.geojson", driver='GeoJSON')
# targets.to_csv('germany_targets.csv')
# targets.to_pickle('germany_targets.pkl')
