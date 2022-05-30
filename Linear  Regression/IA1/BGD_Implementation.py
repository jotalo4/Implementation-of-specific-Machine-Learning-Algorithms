from multiprocessing.sharedctypes import Value
from this import d
from tkinter.font import names
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.utils import column_or_1d

# Import the Standardized Data as Input Data
input_data = pd.read_csv("C:/Users/samue/Desktop/AI534_2021_slides_and_texts/Assignments/Workspace/Preprocessing/IA1_Normalized_Data.csv")
data_new = pd.DataFrame(input_data)

# Configurations
i_size = 24 # Length of input Vector
y_true = input_data.iloc[:,19]
o_size = len(y_true) # Length of Output Vector# Output vector(price), y
n1 = 5 #  of hidden neurons

# Bias or Intercept Data
bias = input_data.iloc[:,0]

# No of Iterations to train NN or Training examples
N = len(input_data)

# Model Development: Model Training and Selection

# - max. number of iterations (fixed) - epochs
epochs = int(5e3)

# - learning-rate (step-size) selection set
# lrs =  [5e-3, 1e-2, 2e-2, 0.1, 0.5]
# learning rate
stepsize = 1e-1

# - regularization scale size selection set
lregs = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]

# - random weight initialization and biad
W1 = np.random.randn(n1, N)
b1 = np.random.randn(n1,1)
W2 = np.random.randn(o_size, n1)
b2 = np.random.randn(o_size,1)

for i in range(1, N):
   x = input_data(i)
   y = y_true(i)

# Create linear regression object
# LinearRegression fits a linear model with coefficients w = (w1, …, wp) to minimize the residual sum of squares 
# between the observed targets in the dataset, and the targets predicted by the linear approximation.
regressor = LinearRegression()

# Train the model using the training sets
regressor.fit(input_data,y_true)
y_pred = regressor.predict(input_data)

# Model Error using MSE
model_error = mean_squared_error(y_true, y_pred)

# Declaring Variables
# Learning rate = 10,10E(-1),10E(-2),10E(-3),10E(-4),10E(-5),10E(-6)

# Gradient Descent for MSE--- L(w) = 1/N (Yi-yi)^2 where i = 1 to N

