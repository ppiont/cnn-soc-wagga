#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 18:32:01 2020.

@author: peter
"""
import numpy as np


def min_max(array, axis=(0, 1, 2)):
    """Min-max scale an array along channels or features.

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
    x_mins = array.min(axis=axis)
    # Compute the maxs along hte axis
    x_maxs = array.max(axis=axis)
    # Get the difference
    diff = x_maxs-x_mins
    # Get index of irrelevant features (max == min)
    zero_channels_idx = np.where(diff == 0)[0]
    # Remove irrelevant features
    array = np.delete(array, zero_channels_idx, -1)
    diff = np.delete(diff, zero_channels_idx)
    x_maxs = np.delete(x_maxs, zero_channels_idx)
    x_mins = np.delete(x_mins, zero_channels_idx)

    # Compute and return the array normalized to 0-1 range.
    print(f'Removed from axis: {zero_channels_idx}')
    return (array-x_mins)/diff
