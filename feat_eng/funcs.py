#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 18:32:01 2020.

@author: peter
"""
import numpy as np


def min_max(array, axis=(0, 1, 2), rm_no_variance=True):
    """Min-max scale an array and optionally remove features with no variance.

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
    zero_channels_idx : ndarray, depending on `rm_no_variance`
        The idx of zero variance features/channels that were removed.

    """
    # Compute the mins along the axis
    x_mins = array.min(axis=axis)
    # Compute the maxs along hte axis
    x_maxs = array.max(axis=axis)
    # Get the difference
    diff = x_maxs-x_mins

    if rm_no_variance is True:
        # Get index of features with no variance (max == min)
        zero_channels_idx = np.where(diff == 0)[0]
        # Remove features with no variance
        array = np.delete(array, zero_channels_idx, -1)
        diff = np.delete(diff, zero_channels_idx)
        x_maxs = np.delete(x_maxs, zero_channels_idx)
        x_mins = np.delete(x_mins, zero_channels_idx)
        print(f'Removed from axis: {zero_channels_idx}')

        # Compute and return the array normalized to 0-1 range and
        # optionally return zero variance indexes
        return (array-x_mins)/diff, zero_channels_idx

    else:
        return (array-x_mins)/diff


def crop_center(arr, crop_x, crop_y):
    """Crop an array, maintaining center."""
    if len(arr.shape) > 3 and np.shape[0] == 1:
        arr = np.squeeze(arr)
    y, x, c = arr.shape
    start_x = x//2 - crop_x//2
    start_y = y//2 - crop_y//2

    return arr[start_y:start_y+crop_y, start_x:start_x+crop_x, :]
