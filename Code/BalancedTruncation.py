import numpy as np
import scipy as sp
import control as ct
import csv
import matplotlib.pyplot as plt

# Read data containing all entries of the state-space matrices
def ReadData(filepath):
    with open(filepath, 'r') as file:
        AllData = list(csv.reader(file, delimiter=';'))

    # Assign variables
    m = int(AllData[1][0])  #input order
    p = int(AllData[1][1])  #output order
    n = int(AllData[1][2])  #problem order
    
    A = np.matrix(AllData[3:n+3], dtype = np.double)
    B = np.matrix(AllData[n+4:2*n+4], dtype = np.double)
    C = np.matrix(AllData[2*n+5:3*n+5], dtype = np.double)
    D = np.matrix(AllData[3*n+6:3*n+p+6], dtype = np.double)
    
    return m, p, n, A, B, C, D
    
# Assigns all needed variables from the provided data
def AssignVars(reducedOrder, A=None, B=None, C=None, D=None, filepath=None):
    r = reducedOrder
    if filepath != None:
        m, p, n, A, B, C, D = ReadData(filepath)
    else:
        if A.all() == None or B.all() == None or C.all() == None or D == None:
            print("Matrices not defined correctly.")
        A = np.matrix(A, dtype=np.double)
        B = np.matrix(B, dtype=np.double).transpose()
        C = np.matrix(C, dtype=np.double).transpose()
        D = np.matrix(D, dtype=np.double)
        m = B.shape[1]
        p = C.shape[1]
        n = A.shape[0]
    return r, m, p, n, A, B, C, D

# Applies a similarity transform to a system
def SimilarityTransform(A, B, C, T, T_inv):
    A_prime = np.matmul(np.matmul(T_inv, A), T)
    B_prime = np.matmul(T_inv, B)
    C_prime = np.matmul(C, T)
    return A_prime, B_prime, C_prime

# This returns the sections of the matrices used in the reduced system based on reduction order r.
def TruncateSystemMatrices(A, B, C, r, m, p):   
    A11 = np.ndarray([r, r])
    for i in range(r):
        for j in range(r):
            A11[i,j] = A[i,j]
            
    B1 = np.ndarray([r, m])
    for i in range(r):
        for j in range(m):
            B1[i,j] = B[i,j]
            
    C1 = np.ndarray([p, r])
    for i in range(p):
        for j in range(r):
            C1[i,j] = C[i,j]
            
    return A11, B1, C1

# Calculates the error-bound corresponding to a certain reduction.
def CalculateBTErrorBound(HSV, r):
    result = 0.0
    for i in range(r, HSV.size):
        result += HSV[i]
    return 2*result
    
# Plot the size of Hankel Singular Values and the error bound for each order of truncation.
def PlotData(HSV):
    fig, (ax_1, ax_2) = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
    
    ax_1.plot(HSV, 'bs')
    ax_1.set_title('Size of the Hankel Singular Values', fontsize='medium')
    ax_1.axis([0, HSV.size, 0, int(HSV[0]) + 1])
    ax_1.set_xlabel('order, n')
    ax_1.set_ylabel('HSV, $\sigma_i$')
    print('HSV')
    print(HSV)
    
    errorbounds = []
    for i in range(HSV.size):
        errorbounds.append(CalculateBTErrorBound(HSV, i))
        
    ax_2.plot(errorbounds, 'rs')
    ax_2.set_title('Size of the BT error bound', fontsize='medium')
    ax_2.axis([0, len(errorbounds), 0, int(errorbounds[0]) + 1])
    ax_2.set_xlabel('order of reduced model')
    ax_2.set_ylabel('Error Bound')
    print('Error bounds')
    print(errorbounds)
    
    fig.suptitle('Balanced Truncation', fontsize='medium')
    plt.show()

    
# Applies Balanced Truncation to the matrices of a state-space system and reduces it to a given order.    
def ApplyBalancedTruncation(SamplingInterval, reducedOrder, A=None, B=None, C=None, D=None, filepath=None):
    r, m, p, n, A, B, C, D = AssignVars(reducedOrder, A, B, C, D, filepath)
    
    BBT = np.matmul(B, B.transpose())
    CT = C.transpose()
    
    DiscreteSystem = ct.StateSpace(A, B, CT, D, SamplingInterval)
    
    # We calculate the cholesky-factors of both Gramians of the system and the Gramians itself.
    Rc = ct.gram(DiscreteSystem, 'cf')
    Wc = np.matmul(Rc, Rc.transpose())
    Ro = ct.gram(DiscreteSystem, 'of')
    Wo = np.matmul(Ro.transpose(), Ro)
    
    T, T_inv, HSV = SquareRootAlgorithm(Rc, Wc, Ro, Wo)

    PlotData(HSV)
    
    A_prime, B_prime, C_prime = SimilarityTransform(A, B, CT, T, T_inv)
    A11, B1, C1 = TruncateSystemMatrices(A_prime, B_prime, C_prime, r, m, p)
    
    return A11, B1, C1
    
#[1] This method  produces a balancing transformation and the Hankel Singular Values
# based on the square root algorithm as stated in Antoulas A.C., Sorensen D.C., "APPROXIMATION OF LARGE-SCALE
# DYNAMICAL SYSTEMS: AN OVERVIEW", Int. J. Appl. Math. Comput. Sci., 2001, Vol.11, No.5, 1093–1121.
def SquareRootAlgorithm(Rc, Wc, Ro, Wo):
    # Setting variables to follow the exact notation in [1].
    P = Wc
    U = Rc
    UT = Rc.transpose()
    
    Q = Wo
    L = Ro.transpose()
    LT = Ro
    
    UTL = UT@L
    Z, S, YT = np.linalg.svd(UTL)
    
    S12_inv = S.copy()
    for i in range(S.size):
        S12_inv[i] = 1/(np.sqrt(S[i]))
    
    Tb = np.diag(S12_inv) @ YT @ LT
    Tb_inv = U @ Z @ np.diag(S12_inv)
    
    HSV = S
    T = Tb_inv
    T_inv = Tb
    
    return T, T_inv, HSV

    