# somefuno@oregonstate.edu
# Read and Preprocess CSV data: (train/dev/test)

import os
import pathlib
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# expects:

# raw data
# normalize
# is it train data.
# train data statistics
# skip feature enigneering

# normalize: age, annual premium and vintage

def reorder_columns(columns, first_cols=[], last_cols=[], drop_cols=[]):
    columns = list(set(columns) - set(first_cols))
    columns = list(set(columns) - set(drop_cols))
    columns = list(set(columns) - set(last_cols))
    new_order = first_cols + columns + last_cols
    return new_order

# outputs
# processed raw data

# Read and Preprocess CSV data: (train/dev/test)
def preprocess(rawdata, donormalize=0, istrain=1, traininfo=None, doengr=0):
    dataframe = pd.read_csv(rawdata)
    outfeature = 'Response'

    print('data size (rows,columns)', dataframe.shape)
    # print(dataframe.head())
    # print(dataframe.describe())
    data_id = dataframe.index

    dfcpy = dataframe.copy()
    # print(dataframe.columns)

        
    

    # Extract X-Y parts
    try:
        Yout = dfcpy[outfeature].to_numpy().reshape(dfcpy.shape[0], 1)
    except KeyError as ke:
        print('Key error [price]: No price-column in test-data')
        Yout = []

    if len(Yout) != 0:
        Xin = dfcpy.iloc[:, 0:dfcpy.shape[1]-1]
    else:
        # for test-data where output feature is not included in the raw dataset
        Xin = dfcpy.iloc[:, 0:dfcpy.shape[1]]

    # for the select fetures: make a copy of the names
    feats_name = list(Xin.columns)
    Xout = Xin.to_numpy().reshape(Xin.shape[0], Xin.shape[1])
    print(Xout)
    if len(Yout) == 0:
        print(feats_name)

    if (donormalize==1) and (numels > 0):
        indata = {'X': Xout, 'Y': Yout,
                  'rows': Xin.shape[0], 'cols': Xin.shape[1],
                  'scalers': scalers, 'feats': feats_name}
    else:
        indata = {'X': Xout, 'Y': Yout,
                  'rows': Xin.shape[0], 'cols': Xin.shape[1],
                  'scalers': None, 'feats': feats_name}

    return indata, data_id


# Helper Functions

def unique(a):
    # returns the unique list of an iterable object
    return list(set(a))


def intersect(a, b):
    # returns the intersection list of two iterable objects
    return list(set(a) & set(b))


def union(a, b):
    # returns the union list of two iterable objects
    return list(set(a) | set(b))


def differ(a, b):
    # returns the list of elements in the left iterable object
    # not in the right iterable object
    return list(set(a) - set(b))
