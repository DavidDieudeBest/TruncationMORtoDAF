import numpy as np
import csv

# Read data containing de coëfficients of the numerator and denominator of a transfer function.
def ReadData(filepath):
    with open(filepath, 'r') as file:
        AllData = list(csv.reader(file, delimiter=';'))
        
    num = [float(i) for i in AllData[0]]
    denom = [float(i) for i in AllData[1]]
        
    # We fill coëfficients for the highest powers with zeros
    # if one of the list contains less elements than the other.
    if len(num) > len(denom):
        for i in range(len(denom), len(num)):
            denom.append(0)
    else:
        for i in range(len(num), len(denom)):
            num.append(0)
     
    return num, denom

#[1] These methods convert a transfer function to a state-space model in controller canonical form 
# using the method described by Smith, J.O., Introduction to Digital Filters. 
# [Online] http://ccrma.stanford.edu/~jos/filters/, Sept. 2005, online book. Accessed in April 2020.

# We extract a delay-free path to find the feedthrough gain
# used as the direct-path coëfficient in the state-space model
def ExtractDelayFreePath(num, denom):
    if num[0] == 0:
        return num, denom
    else:
        for i in range(len(num) - 1):
            num[i] = num[i] - num[-1] * denom[i]
    return num, denom

# We construct the state-space matrices from the transfer function.
def FillMatrices(num, denom):
    order = len(num) - 1
    A = np.zeros((order, order))
    B = np.zeros(order)
    C = np.zeros(order)
    D = num[0]
    
    # We construct a model in the controller canonical form defined in [1].
    for i in range(order):
        A[0][i] = -denom[order - i - 1]
        if i > 0:
            A[i][i-1] = 1
        
        C[i] = num[order - i - 1]
        
    B[0] = 1
        
    return A, B, C, D
    
# Converts a transfer function to state-space model.
def ConvertToStateSpace(num=None, denom=None, filepath=None):
    if filepath != None:
        num, denom = ReadData(filepath)
    if num.all() == None or denom.all() == None:
        print("Numerator or denominator not assigned.")
    num, denom = ExtractDelayFreePath(num, denom)
    return FillMatrices(num, denom)
            
