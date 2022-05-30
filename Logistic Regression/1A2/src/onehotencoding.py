# samuelayomikun@gmail.com
# Read and Preprocess CSV data: (train/dev/test)

import os
import pathlib
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder


# Encode categorical features as a one-hot numeric array.
def onehotencoder(rawdata):
    
    data = pd.read_csv(rawdata)
    df = pd.DataFrame(data)
    # Identify categorical features
    categorical_features =  ['Previously_Insured', 'Vehicle_Damage', 'dummy']
    # create an instance of one-hot-encoder
    encoder = OneHotEncoder(n_values='auto', handle_unknown='ignore') 
    # Perform one-hot encoding on specific column
    
    
    # Transform Data
    encoded_DT = encoder.fit_transform(df[['categorical_features']])
    # encoder dataframe
    encoder_df = pd.DataFrame(encoded_DT.toarray())
    # Merger one-hot encoded columns back with original dataframe
    final_df = df.join(encoder_df)
    return(encoder_df, final_df)

def encode_and_bind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    res = pd.concat([original_dataframe, dummies], axis=1)
    res = res.drop([feature_to_encode], axis=1)
    return(res)
                