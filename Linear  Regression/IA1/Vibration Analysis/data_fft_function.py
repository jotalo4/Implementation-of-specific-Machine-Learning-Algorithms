# Written by: Orukotan, Ayomikun Samuel
# Signal Processing. Time Domain --> Frequency Domain
# expects: Input Data in time series 
# X.shape: i x j,
# W.shape = j x k

# returns: Y.shape: i x k
import numpy as np
import pandas as pd
from numpy.fft import fft, ifft
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def preprocess(rawdata):
    dataframe = pd.read_csv(rawdata)
    #dataframe = dataframe.to_numpy()
    # print(dataframe.head())
    # print(dataframe.describe())
    #find minimum and maximum values of array
    #min_val = np.min(inputdata)
    #max_val = np.max(inputdata)
    
    data_id = dataframe.iloc[:,0]
    cols = dataframe.columns
    columns = list(dataframe)
    datac = dataframe.drop(columns= ['Position','Direction','Point','Units','% Change'])
    inputdata = datac
    colum = list(inputdata)
    print(colum)
    inputdata = inputdata.copy()
    print('data size (rows,columns)', inputdata.shape)
    return [data_id, cols, inputdata]

def fftprocess(rawdata):
    dataframe = pd.read_csv(rawdata)
    #dataframe = dataframe.to_numpy()
    # print(dataframe.head())
    # print(dataframe.describe())
    #find minimum and maximum values of array
    #min_val = np.min(inputdata)
    #max_val = np.max(inputdata)
    
    data_id = dataframe.iloc[:,0]
    cols = dataframe.columns
    columns = list(dataframe)
    datac = dataframe.drop(columns= ['Position','Direction','Point','Units', 'Collection Date','% Change'])
    inputdata = datac
    colum = list(inputdata)
    print(colum)
    inputdata = inputdata.copy()
    print('data size (rows,columns)', inputdata.shape)
    return [data_id, cols, inputdata]

def save_multi_image(filename):
    pp = PdfPages(filename)
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]
    for fig in figs:
      fig.savefig(pp, format='pdf')
    pp.close()
    return(filename)
#filename = ("Vibration Analysis\figs\VibrationAnalysis")
#save_multi_image(filename)



# extracting input data 
def fourier(inputdata, Fs):
   N = len(inputdata) # Length of Signal
   Fs = 2e3 # Sampling Rate
   # window length
   n = np.arange(N) # returns array of evenly spaced values
   T = N/Fs
   #freq = n/T 
   
#    # Maximum Frequency of the Signal
#    Fmax = np.max(X)
#    # Sampling Frequency 
#    Fs = 2*Fmax
   # Sampling rate using Nyquist Criterion, Fs = 2Fmax; Fmax = 1/T
   # time vector
#    t = np.arange(0, n/Fs, 1/Fs)
   
   # fft of X 
   # Y = fft(X) computes the discrete Fourier transform (DFT) of X using a fast Fourier transform (FFT) algorithm.
   X = fft(inputdata)
   # Inverse of FFT
   Y = ifft(X)
   
   # Return the Discrete Fourier Transform sample frequencies.
   freq = np.fft.fftfreq(N)
   
   
   # return Output
   return[X,Y,freq,N]


# Helpful Functions
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

# a = current_value b = previous_value. (New-old)/old * 100
def perct_change(a, b):
    percntInc = ((a-b) / b) * 100
    return(percntInc)
    
    
