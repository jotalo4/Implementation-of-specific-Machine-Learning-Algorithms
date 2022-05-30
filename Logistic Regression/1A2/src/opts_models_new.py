# Written by Orukotan, Ayomikun Samuel
# Gradient-Based Optimization or Learning API

import sklearn.metrics
from src.infer_models import infer
from src.archs_models import *
import numpy as np
import pandas as pd
np.set_printoptions(precision=4)


# expects:
# modeldictionary,
# model,
# traindata,
# devdata

# returns:
# mse train, and  mse validation
# model dictionary

# gradient is the function or any Python callable object that takes a vector and returns the gradient of the function you’re trying to minimize.
# learn_rate is the learning rate that controls the magnitude of the vector update.
# n_iter is the number of iterations.
# start is the point where the algorithm starts its search, given as a sequence (tuple, list, NumPy array, and so on) 
# Gradient Descent Function: Here iterations, learning_rate, stopping_threshold are hyperparameters that can be tuned
def gradient_descent(modeldict, model, traindata, devdata):
    
    mse = []
    faccs = []
    paccs = []
    stopping_threshold = 1e-6
    mse_val = []
    facc_val = []
    pacc_val = []

    # extract
    batchlen = traindata["rows"]
    X = traindata["X"]
    Y = traindata["Y"]
    W = modeldict["W"]
    lambda_lr = modeldict["stepsize"]
    regula = modeldict["reg_type"]
    lambda_reg = modeldict["reg_size"]
    epochs = modeldict["epochs"]
    previous_cost = None

    # Estimation of optimal parameters
    for k in range(epochs):

        # predict values of Y
        Yhat = model(X, W)
        
        # error
        e = Y - Yhat

        # decision boundary ~ 0.5
        # y_hat(y_hat >= 0.5) = 1;
        # y_hat(y_hat < 0.5) = 0;
        decbnds = 0.5
        Yhat = np.where(Yhat > decbnds, 1, 0)
        # error
        # e = Y - Yhat

        # accuracy. accuracy = correct predictions/all predictions
        num_correct = np.sum(Yhat == Y)
        facc = num_correct/batchlen
        pacc = facc*100

        # mse: cost, Calculating the current Cost
        error_cost = np.sum(np.square(e))/batchlen
        
        mse.append(error_cost) 
        faccs.append(facc)
        paccs.append(pacc)
        
        # If the change in cost is less than or equal to stopping_threshold we stop the gradient descent
        if previous_cost and abs(previous_cost-error_cost)<=stopping_threshold:
            break
        previous_cost = error_cost
                
        # GD: Batch

        # output jacobian
        J = X

        # first-order gradient of the cost-function
        # g = np.dot(J.T,e)
        g = (J.T)@(e)

        # step size multiplied with descent direction
        p = (lambda_lr/batchlen)*(np.pi*g)
        # print(np.size(p))

        # weight change
        dW = p

        # - update
        W = W + dW
        # Regularize, to avoid overfitting
        # lambda is a Hyperparameter Known as regularization constant
        # regularize weights and de-regularize bias
        if regula == 2:
            W = W - (lambda_lr*lambda_reg)*W
            W[0] = W[0] + (lambda_lr*lambda_reg)*W[0]
        elif regula == 1:
            W -= (lambda_lr*lambda_reg)*np.sign(W)
            W[0] += (lambda_lr*lambda_reg)*np.sign(W[0])
        elif regula == 3:
            aw = 0.2 # [0 1]
            elasticW = (1-aw)*np.sign(W) + (aw*W)
            W -= (lambda_lr*lambda_reg)*(elasticW)
            W[0] += (lambda_lr*lambda_reg)*elasticW[0]
        elif (regula != 2) or (regula != 1) or (regula != 3):
            W = W
        
            # do nothing
        

        modeldict["W"] = W
        [Yhatv, msev, faccv, paccv] = infer(model, modeldict, devdata)
        mse_val.append(msev)
        facc_val.append(faccv)
        pacc_val.append(paccv)
        
        # log progress, every 400 batches.
        if k % 200 == 0:
            messagelog = (
                f"k: {k:5d}, mse(train): {mse[k]:2.4f}, mse(dev): {mse_val[k]:2.4f} | "
                f"facc(train): {faccs[k]:2.4f}, facc(dev): {facc_val[k]:2.4f}"
            )
            print(messagelog)
            
      # - compute weight sparsity for this model
    sparse_num = np.sum(W <= 1e-6)
    # print(f"Model Sparsity: {sparse_num}")
    # store
    modeldict['sparsity'] = sparse_num
    
     # - find top 5 features with the largest magnitude
    # # sort in descending order
    feats_idx = (-np.abs(W)).argsort(axis=0).T
    bigk = modeldict['bigk'] 
    # #get big 5 or big k  ids as may be the case
    feats_idx_bigK = feats_idx[0,:bigk]
    # #get corresponding features
    feats_bigKwts = [traindata['feats'][idx] for idx in feats_idx_bigK]
    feats_bigKwts = np.array(traindata["feats"])[feats_idx_bigK]
    WBigK = W[feats_idx_bigK].T
    WBigK_feats = feats_bigKwts
    # print(f"(Big-5) Learned Weights: {WBigK}")
    # print(f"(Big-5) Corresponding Features: {WBigK_feats}")
    # #store
    modeldict["WBigK"] = WBigK
    modeldict['WBigK_feats'] = WBigK_feats

    # store model params
    modeldict["W"] = W

    # - store metrics
    modeldict['error_list'] = error_cost
    modeldict['mse_train'] = mse
    modeldict['mse_dev'] = mse_val
    modeldict['facc_train'] = faccs
    modeldict['facc_dev'] = facc_val
    #modeldict['epochs'] = k

 
 


