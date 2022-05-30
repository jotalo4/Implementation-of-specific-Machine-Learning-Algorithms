# somefuno@oregonstate.edu
# Architectures, Models, Hypothesis Spaces API
# Activation Functions

import numpy as np

# - Linear Model: linear(X,W)
# expects: 
# X.shape: i x j,
# W.shape = j x k

# returns: Y.shape: i x k
# Hyperbolic or tanh Function
def tanh(X):
    t=(np.exp(X)-np.exp(-X))/(np.exp(X)+np.exp(-X))
    dt=1-t**2
    return t,dt

# Linear Function
def linear(X,W):
    return X.dot(W) # 
    #return np.einsum('ij,jk->ik',X,W);


# - Standard Logistic Model: stdlogistic(X,W)
def stdlogistic(X,W):
    #v = X.dot(W) 
    #v = np.einsum('ij,jk->ik',X,W)
    v = X@W
    return  1/(1 + np.exp(-v))

# Sigmoid Function
def sigmoid(X):
    s=1/(1+np.exp(-X))
    ds=s*(1-s)  
    return s,ds


# Relu Function
def ReLU(X):
    data = [max(0,value) for value in X]
    return np.array(data, dtype=float)

# Derivative of Relu Function
def der_ReLU(X):
    data = [1 if value>0 else 0 for value in X]
    return np.array(data, dtype=float)
 
 # Softmax function
def softmax(X):
     expo = np.exp(X)
     expo_sum = np.sum(np.exp(X))
     return expo/expo_sum



