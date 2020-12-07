#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 17:07:53 2020.

@author: peter
"""
import numpy as np


def mean_error(y_true, y_pred):
    """Calculate the mean error."""
    assert len(y_true) == len(y_pred)
    return np.mean(y_pred-y_true)


def lin_ccc(y_true, y_pred):
    """Calculate Lin's concordance correlation coefficient."""
    cor = np.corrcoef(y_true, y_pred)[0][1]
    numerator = 2 * cor * np.std(y_true) * np.std(y_pred)
    denominator = np.var(y_true) + np.var(y_pred) + (np.mean(y_true)
                                                     - np.mean(y_pred))**2
    return numerator/denominator
