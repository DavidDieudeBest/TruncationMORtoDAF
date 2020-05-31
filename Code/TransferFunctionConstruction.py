import numpy as np
import csv

import matplotlib.pyplot as plt

# [1] This method constructs a transfer function numerator and denominator based on the method
# presented by Abel, J.S. and Smith, J.O., "ROBUST DESIGN OF VERY HIGH-ORDER ALLPASS DISPERSION FILTERS",
# in Proc. of the 9th Int. Conference on Digital Audio Effects (DAFx-06), Montreal, Canada, September 18-20, 2006.
def ConstructTransferFunction(filepath):
    # Read the given bounds of the area bands of the group delay.
    with open(filepath, 'r') as file:
            AllData = list(csv.reader(file, delimiter=';'))
    bounds = [float(i) for i in AllData[0]]
    
    # This variable determines the overlap of successive band group delays,
    # used to determine the smoothness of the approximation
    OverlapBeta = 0.9
    SamplesPerSecond = bounds[-1] * 2
    SamplingInterval = 1 / SamplesPerSecond
    order = len(bounds) - 1
    
    # We calculate a normalized angular frequency for each bound using double the Nyquist limit.
    NormAngFreq = [2*np.pi*i / SamplesPerSecond for i in bounds]

    BandMidpoints = np.zeros(order)
    # Intermediate variables
    Deltas = np.zeros(order)
    Etas = np.zeros(order)
    BetaPrime = np.sqrt(OverlapBeta / (1 - OverlapBeta))
    # The pole radius is used to define all-pass factored biquad sections of the transfer function.
    PoleRadius = np.zeros(order)
    # This radius is assigned using an approximation, that can be used for large filters.
    PoleRadius2 = np.zeros(order)
    
    # Following the calculations done in [1].
    for i in range(order):
        BandMidpoints[i] = (NormAngFreq[i] + NormAngFreq[i+1]) / 2
        Deltas[i] = (NormAngFreq[i+1] - NormAngFreq[i]) / 2
        Etas[i] = (1 - OverlapBeta * np.cos(Deltas[i])) / (1 - OverlapBeta)
        PoleRadius[i] = Etas[i] - np.sqrt(Etas[i]**2 - 1)
        PoleRadius2[i] = 1 - BetaPrime * Deltas[i]
        
    PoleFrequency = BandMidpoints
    
    TransferNumerator = np.poly1d(1)
    TransferDenominator = np.poly1d(1)
    
    # We calculate the numerator and denominator of the final transfer function separately using polynomial multiplication. 
    for i in range(order):
        # We add only components to the transfer function that will yield a stable system.
        if PoleFrequency[i] >= 0.5 * np.pi or PoleFrequency[i] <= -0.5 * np.pi:
            a = PoleRadius[i]**2
            b = -2 * PoleRadius[i] * np.cos(PoleFrequency[i])
            TransferNumerator = TransferNumerator * np.poly1d([1, b, a])
            TransferDenominator = TransferDenominator * np.poly1d([a, b, 1])

    # We return only the coëfficients in reverse order in a two dimensional array.
    result = []
    result.append(np.asarray(TransferNumerator))
    result.append(np.asarray(TransferDenominator))
    
    PlotGroupDelay(PoleRadius, PoleFrequency, order)
    return result, SamplingInterval

# This method plots the group delay that is imposed by a filter with 
# the given pole radii and pole frequencies and also plots the group delay imposed 
# by each individual all-pass section of the filter transfer function.
def PlotGroupDelay(PoleRadius, PoleFrequency, order):
    ydata = []
    xdata = []
    allydata = []
    for j in range(order):
        subydata = []
        for i in range(1000):
            subydata.append((1-PoleRadius[j]**2)/(1+PoleRadius[j]**2-2*PoleRadius[j]*np.cos(i*0.001*np.pi-PoleFrequency[j])))
        allydata.append(subydata)
    for i in range(1000):
        xdata.append(i*0.001*np.pi)
        sum = 0
        for j in range(len(allydata)):
            sum += allydata[j][i]
        ydata.append(sum)


    plt.plot(xdata, ydata)
    for i in range(order):
        plt.plot(xdata, allydata[i], 'g') 

    plt.xlim([0, np.pi])
    plt.xlabel('Frequency, $\omega$')
    plt.ylabel('Group delay, $\delta(\omega)$')
    plt.show()