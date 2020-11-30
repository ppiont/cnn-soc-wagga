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
import os

# Go to data folder if not already in it
if os.getcwd().split(r"/")[-1] != "data":
    os.chdir("data/")


def minmax_array(array, axis=(0, 1, 2)):
    """Min-max scale an array along some axis or axes.

    Parameters
    ----------
    array : ndarray
        Array to be scaled.
    axis : int or tuple of int, optional
        The axis or axes along which to scale. The default is (0, 1, 2).

    Returns
    -------
    ndarray
        The scaled array.

    """
    # Compute the mins along the axis
    x_mins = numerical.min(axis=axis)
    # Compute the maxs along hte axis
    x_maxs = numerical.max(axis=axis)

    # Compute and return the array normalized to 0-1 range.
    return (array-x_mins)/(x_maxs-x_mins)


# --------------------------- Read and prep data --------------------------- #
data = pd.read_pickle("targs_feats_IDs.pkl")

train, test = train_test_split(data, test_size=0.1, random_state=43)

train_x = np.stack(train["features"])
train_y = train["SOC"]

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                                        featurewise_center=True,
                                        featurewise_std_normalization=True)

datagen.fit(train_x)

it = datagen.flow(train_x, train_y, batch_size=32)

# categorical_features = np.argwhere(np.array(
# [len(set(train[:,:,:,x])) for x in range(train.shape[3])]) <= 5).flatten()


# ----------------------- Define Neural Network Class ----------------------- #
class neural_net(tf.keras.Model):
    """ Summary or description of the function/class.

    Parameters
    ----------

    Returns
    -------

    """

    def __init__(self):
        super(neural_net, self).__init__()

        # # Define lookback
        # self.lookback = lookback
        # # Define dropout
        # self.dropout = dropout
        # # Define r_dropout
        # self.r_dropout = r_dropout

        # Define CNN block
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3),
                                            activation=tf.nn.relu)
        self.bnorm1 = tf.keras.layers.BatchNormalization()
        self.maxpool1 = tf.keras.layers.MaxPooling2D((2, 2))
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3),
                                            activation=tf.nn.relu)
        self.bnorm2 = tf.keras.layers.BatchNormalization()

        # Flatten
        self.flatten = tf.keras.layers.Flatten()

        # Define dense layers
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.dense3 = tf.keras.layers.Dense(1, activation='linear')

    # Define the forward propagation
    def call(self, inputs):

        # Run CNN layers
        x = self.conv1(inputs)
        x = self.bnorm1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.bnorm2(x)

        # Flatten
        x = self.flatten(x)

        # Run dense layers
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)

        # Return output
        return x


# Create an instance of neural network model
model = neural_net()

# Define optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# Define loss function
loss_fn = tf.keras.losses.MeanSquaredError()

# Compile model
model.compile(optimizer=optimizer, loss=loss_fn)

# Define callbacks
# early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
# learning_rate_reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

# Metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')
val_loss = tf.keras.metrics.Mean(name='val_loss')


model.fit(it, steps_per_epoch=len(train) / 32, epochs=15)
