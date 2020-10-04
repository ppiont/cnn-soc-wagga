#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 14:51:29 2020

@author: peter
"""

import os
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import statistics
import numpy as np
import scipy.stats

mpl.style.use('tableau-colorblind10')
mpl.rcParams.update({"lines.linewidth": 1, "font.family": "serif", 
                     "xtick.labelsize": "small", "ytick.labelsize": "small", 
                     "xtick.major.size" : 0, "xtick.minor.size" : 0, 
                     "ytick.major.size" : 0, "ytick.minor.size" : 0, 
                     "axes.titlesize":"medium", "figure.titlesize": "medium", 
                     "figure.figsize": (5, 5), "figure.dpi": 600, 
                     "figure.autolayout": True, "savefig.format": "pdf", 
                     "savefig.transparent": True, "image.cmap": "viridis"})

pd.options.display.precision = 2

if os.getcwd().split(r"/")[-1] != "data":
    os.chdir("data/")
    
fig_path = "../figures/"

df = pd.read_csv("germany_targets.csv", index_col = 0)
oc = df.OC

fig, ax = plt.subplots()
ax.hist(df["OC"], bins = 50)
# plt.figtext(0.6,0.5, df["OC"].describe().to_string())
plt.figtext(0.7,0.75, 
            f"$N={len(oc)}$\n$\\mu={round(oc.mean(), 2)}$\n$\\sigma={round(oc.std(), 2)}$\n$min={round(oc.min(), 2):.2f}$\n$max={round(oc.max(), 2):.2f}$", 
            fontsize = 'large')
plt.xlabel("SOC (g/kg)")
plt.ylabel("Count")
plt.savefig(os.path.join(fig_path, "sample_histogram.pdf"), bbox_inches = 'tight', pad_inches = 0)
plt.show()