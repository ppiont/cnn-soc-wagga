#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 14:43:26 2020

@author: peter
"""

import owslib
from owslib.wms import WebMapService
import gdal


wms = WebMapService('https://maps.isric.org/mapserv?map=/map/ocs.map', version='1.3.0')

wms['ocs_0-30cm_mean'].crsOptions
bbox_germany = (5.98865807458, 47.3024876979, 15.0169958839, 54.983104153)

img = wms.getmap(layers=['ocs_0-30cm_mean'],
                 srs = 'EPSG:4326',
                 bbox = bbox_germany,
                 size = (400,300),
                 format ='image/png',
                 transparent = True
                 )

with open('germany_ocs_0-30cm_mean', 'wb') as out:
    out.write(img.read())
