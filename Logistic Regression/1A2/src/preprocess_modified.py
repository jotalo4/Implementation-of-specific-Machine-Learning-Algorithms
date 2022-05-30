# somefuno@oregonstate.edu
# Read and Preprocess CSV data: (train/dev/test)

# importing libraries
import os
import pathlib
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas
import numpy


# summarize transformed data
np.set_printoptions(precision = 4)
   

# expects:

# raw data
# normalize
# is it train data.
# train data statistics
# skip feature enigneering

# normalize: age, annual premium and vintage

# outputs
# processed raw data

# Read and Preprocess CSV data: (train/dev/test)
def preprocess(rawdata, donormalize=0):
    dataframe = pd.read_csv(rawdata)
    outfeature = 'Response'
    print('data size (rows,columns)', dataframe.shape)
    data_id = dataframe.index
    dfcpy = dataframe.copy()
    
    # Normalize only selected: numerical features to normal dists.
    normfeats = ['Age', 'Annual_Premium', 'Vintage']
    
# account for dropped features, if any
    allfeats = list(dfcpy.columns)
    allfeats = allfeats[:-1]
    normfeats = intersect(normfeats, allfeats)
    numels = len(normfeats)
    
    # separate dataframe array into input and output components
    # for test-data where output feature is not included in the raw dataset
    Xin = dfcpy.iloc[:, 0:dfcpy.shape[1]]
    feats_name = list(Xin.columns) # for the select fetures: make a copy of the names
    
    Xout = Xin.to_numpy().reshape(Xin.shape[0], Xin.shape[1]) # X-Data converted to numpy array
    Yout = dfcpy[outfeature].to_numpy().reshape(dfcpy.shape[0], 1)  # Y-Data converted to numpy array
    
    scaler = StandardScaler().fit(Xout) # define standard scaler
    rescaledX = scaler.transform(Xout)  # transform data
    print(rescaledX)
     
    if (donormalize==1) and (numels > 0):
        indata = {'X': Xout, 'Y': Yout,
                  'rows': Xin.shape[0], 'cols': Xin.shape[1],
                  'scalers': rescaledX, 'feats': feats_name}
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
