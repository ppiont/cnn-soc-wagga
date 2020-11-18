#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 16:27:30 2020

@author: peter
"""

from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import numpy as np

data = pd.read_pickle("targs_feats_IDs.pkl")

train, test = train_test_split(data, test_size = 0.1, random_state = 43)

datagen = tf.keras.preprocessing.image.ImageDataGenerator()

datagen.flow(train.iloc[:,1])

# testarray = train.iloc[:,1]
# testarray[0].shape

# train.features[0].shape

# test2 = np.expand_dims(train.features, 0)
# test2.shape
