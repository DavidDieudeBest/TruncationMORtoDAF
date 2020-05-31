import numpy as np
import scipy as sp
import control as ct
import csv
import matplotlib.pyplot as plt

import TransferFunctionConstruction as tfc
import TransferFunctionToStateSpace as tftoss
import BalancedTruncation as bt
import ModalTruncation as mt

# NOTE: Uncomment one block of code to obtain data for that case.



## Basic example
#A, B, C, D = tftoss.ConvertToStateSpace(np.array([1, -5/2, 1, 1, -1/2]), np.array([1/12, 41/84, 67/42, 47/24, 1]))
#SamplingInterval = 1/8000


## Order 10 filter
TransferFunction, SamplingInterval = tfc.ConstructTransferFunction("../Data/GDBLinear5BQ.csv")
A, B, C, D = tftoss.ConvertToStateSpace(TransferFunction[0], TransferFunction[1])


## Order 10 filter B
#TransferFunction, SamplingInterval = tfc.ConstructTransferFunction("../Data/GDBLinear5BQ_B.csv")
#A, B, C, D = tftoss.ConvertToStateSpace(TransferFunction[0], TransferFunction[1])


## Order 40 filter
#TransferFunction, SamplingInterval = tfc.ConstructTransferFunction("../Data/GDBLinear20BQ.csv")
#A, B, C, D = tftoss.ConvertToStateSpace(TransferFunction[0], TransferFunction[1])


## Order 40 filter B
#TransferFunction, SamplingInterval = tfc.ConstructTransferFunction("../Data/GDBLinear20BQ_B.csv")
#A, B, C, D = tftoss.ConvertToStateSpace(TransferFunction[0], TransferFunction[1])

ReducedOrder = 9
# BalancedTruncation result
A11, B1, C1 = bt.ApplyBalancedTruncation(SamplingInterval, ReducedOrder, A, B, C, D, None)
# Modal Truncation result
A_reduced, B_reduced, C_reduced, residues_reduced, poles_reduced, D = mt.ApplyModalTruncation(ReducedOrder, A, B, C, D, None)

# Plots the impulse response of a given system for a given number of samples.
# First provide the matrices of the original system and then the matrices of the approximation.
# Plotting the impulse response of systems reduced with balanced truncation is possible,
# but systems reduced with modal truncation gives problems as it is a complex system.
# The impulse response of order 10 filter is most interesting for 50 samples,
# but for the order 40 filters the first 30 samples give a useful view.
ImpulseLength = 50
def PlotImpulseResponse(A, B, C, D, A11, B1, C1):
    newB = np.asarray(B).reshape(-1)
    newC = np.asarray(C).reshape(-1)
    newB1 = np.asarray(B1).reshape(-1)
    newC1 = np.asarray(C1).reshape(-1)

    yOriginal = [D]
    xOriginal = []
    xOriginal.append(np.zeros(newB.size))
    xOriginal.append(newB)

    for i in range(1, ImpulseLength):
        xOriginal.append(np.matmul(A, xOriginal[-1]))
        yOriginal.append(np.matmul(newC, xOriginal[i]))
    
    yApprox = [D]
    xApprox = []
    xApprox.append(np.zeros(newB1.size))
    xApprox.append(newB1)
    
    for i in range(1, ImpulseLength):
        xApprox.append(np.matmul(A11, xApprox[-1]))
        yApprox.append(np.matmul(newC1, xApprox[i]))
    
    
    plt.plot(yOriginal, color = 'grey', linestyle = 'dashed')
    plt.plot(yApprox, color = 'blue', linewidth = 2.5)
    plt.show()
    

PlotImpulseResponse(A, B, C, D, A, B, C)  
PlotImpulseResponse(A, B, C, D, A11, B1, C1)
    