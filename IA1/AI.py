from multiprocessing.sharedctypes import Value
from this import d
from tkinter.font import names
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn import preprocessing

# Read CSV file into DataFrame, df
data = pd.read_csv("C:/Users/samue/Desktop/Assignments/Workspace/IA1/csvs/IA1_train.csv")

#splitting the Input data using Pandas Indexing
ID = data.iloc[:,0] 
Date = data.iloc[:,1] 
Bedrooms = data.iloc[:,2] 
Bathrooms  = data.iloc[:,3]
sqft_living = data.iloc[:,4] 
sqft_lot = data.iloc[:,5] 
floors = data.iloc[:,6] 
waterfront = data.iloc[:,7]
view = data.iloc[:,8] 
condition = data.iloc[:,9]
grade = data.iloc[:,10] 
sqft_above = data.iloc[:,11]
sqft_basement = data.iloc[:,12] 
yr_built = data.iloc[:,13] 
yr_renovated = data.iloc[:,14]
zipcode = data.iloc[:,15]
lat = data.iloc[:,16] 
long = data.iloc[:,17] 
sqft_living15 = data.iloc[:,18] 
sqft_lot15 = data.iloc[:,19]

# Output vector, price/100k
price = data.iloc[:,20]

# Splitting Date Features into year, month and day elements
data[["Month","Day","Year"]] = Date.str.split("/", expand = True)

# Adding a dummy feature with a constant value of 1. This helps to learn the intercept or bias term
# data.insert(position,column_name,value)
data.insert(0,'Dummy',1)

#Excluding ID, date features from the original data
data_exc = data.loc[:,~data.columns.isin(['id', 'date'])]

# New Data after exclusing the date feature
data_new = data_exc
data2 = pd.DataFrame(data_new)
age_since_renovated = []

# Constructing a new feature called age_since_renovated
for i, row in data2.iterrows():
    if row["yr_renovated"] == 0:
        row["age_since_renovated"] = pd.to_numeric(row["Year"]) - pd.to_numeric(row["yr_built"] )
        age_since_renovated.append(row["age_since_renovated"])
    else:
        row["age_since_renovated"] = pd.to_numeric(row["Year"]) - pd.to_numeric(row["yr_renovated"])
        age_since_renovated.append(row["age_since_renovated"])
data2["Age_since_renovated"] = age_since_renovated
#print (data2)

# Save the new Data as a csv file
#data2.to_csv("C:/Users/samue/Desktop/Assignments/Workspace/IA1/csvs/IA1_train_modified.csv")
 
#Excluding waterfront from data2
data3 = data2.loc[:,~data2.columns.isin(['Dummy','waterfront', 'price'])]
new_data = data3
data4 = pd.DataFrame(data3)

# Convert DataFrame to a Numpy array of
data5 = data4.to_numpy()

# Save the new Data as a csv file
#data4.to_csv("C:/Users/samue/Desktop/Assignments/Workspace/IA1/csvs/Pre_Standardized_Data.csv")


# Data Standardization using Z-scores. Mean = 0, Standard Deviation = 1. z=(x-mean)/std dev.
# The ultimate reason for standardization is to bring down all the features across columns to a common scale without distorting the differences in the range of the values.
# Define Standard scaler
scaler = preprocessing.StandardScaler()

# Transform Data
scaled_Data = scaler.fit_transform(data5)
Standardized_Data = pd.DataFrame(scaled_Data, columns = data4.columns, index = data4.index)

# Re-insert Dummy for biasing and price as an output element
Standardized_Data.insert(0,'Dummy',1,)
Standardized_Data['wavefront'] = data2.iloc[:,6]
Standardized_Data['price'] = data2.iloc[:,19]
print(Standardized_Data)

# Save the normalized Data as a csv file
Standardized_Data.to_csv("C:/Users/samue/Desktop/Assignments/Workspace/IA1/csvs/IA1_Normalized_Data.csv", index = False)

