import numpy as np
import scipy as sp
import control as ct
import csv
import matplotlib.pyplot as plt

#[1] These methods provide functionality for implementing Modal Truncation as described in 
# Antoulas, A.C. "Model order reduction : methods, concepts and properties".  (CASA-report; Vol. 1507). Eindhoven: Technische Universiteit Eindhoven.(2015). 

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
        B = np.matrix(B, dtype=np.double)
        C = np.matrix(C, dtype=np.double)
        D = np.matrix(D, dtype=np.double)
        m = B.shape[1]
        p = C.shape[1]
        n = A.shape[0]
    return r, m, p, n, A, B, C, D

# This methods creates an array of poles and residues of the transfer function of a given state-space model in pole-residue form.
def CalculatePoleResidue(A, B, C):
    # Following [1], we calculate the eigenvalues and left and right eigenvectors.
    labda, v, w = sp.linalg.eig(A, left = True, right = True)
    residues = []
    scaledLeft = v.copy()
    scaledRight = w.copy()
    
    for i in range(labda.size):
        left = v[:,i]
        right = w[:,i]
        prod = np.vdot(right, left)
        scale = np.sqrt(1/prod)
        scaleconj = np.conj(scale)
        
        newright = np.array(scaleconj * right)
        newleft = np.array(scale * left) 
        newB = np.asarray(B).reshape(-1)
        newC = np.asarray(C).reshape(-1)

        R_i = np.vdot(newC, newright) * np.vdot(newleft, newB)

        residues.append(R_i)
        scaledLeft[:,i] = newleft
        scaledRight[:,i] = newright
        
    poles = labda
    return poles, residues, scaledLeft, scaledRight

# Calculates the dominance of each pole as defined in [1].
def CalculateDominance(poles, residues):
    dominances = []
    for i in range(poles.size):
        dominances.append(np.abs(residues[i])/np.abs(np.real(poles[i])))
    return dominances

# Sorts the poles, residues and dominances of each pole based on the size of the dominances in reverse order.
def SortForDominance(poles, residues, dominances, leftEigVec, rightEigVec):
    poles_sorted = []
    residues_sorted = []
    dominances_sorted = []
    leftEigVec_sorted = leftEigVec.copy()
    rightEigVec_sorted = rightEigVec.copy()
    
    order = np.argsort(dominances)

    for j in range(1, len(poles) + 1):
        poles_sorted.append(poles[order[len(order)- j]])
        residues_sorted.append(residues[order[len(order)- j]])
        dominances_sorted.append(dominances[order[len(order)- j]])
        leftEigVec_sorted[:,j-1] = leftEigVec[:, order[len(order)- j]]
        rightEigVec_sorted[:,j-1] = rightEigVec[:, order[len(order)- j]]

    return poles_sorted, residues_sorted, dominances_sorted, leftEigVec_sorted, rightEigVec_sorted

# Produces the numerator and denominator of the transfer functions reduced to order r.
def ReduceTransferFunction(poles_sorted, residues_sorted, D, r):
    residues_reduced = []
    poles_reduced = []
    
    for i in range(r):
        residues_reduced.append(residues_sorted[i])
        poles_reduced.append(poles_sorted[i])
        
    return residues_reduced, poles_reduced, D

def ReduceEigenVectorMatrices(leftEigVec_sorted, rightEigVec_sorted, n, r):
    leftEigVec_reduced = np.zeros((n, r), dtype = complex)
    rightEigVec_reduced = np.zeros((n, r), dtype = complex)
        
    for i in range(r):
        leftEigVec_reduced[:,i] = leftEigVec_sorted[:,i]
        rightEigVec_reduced[:,i] = rightEigVec_sorted[:,i]
        
    return leftEigVec_reduced, rightEigVec_reduced

def ReduceSystem(A, B, C, leftEigVec_reduced, rightEigVec_reduced):
    A_reduced = leftEigVec_reduced.conj().transpose() @ A @ rightEigVec_reduced
    B_reduced = leftEigVec_reduced.conj().transpose() @ np.asarray(B).reshape(-1)
    C_reduced = C @ rightEigVec_reduced
    return A_reduced, B_reduced, C_reduced

# Calculates the error bound associated with the reduction of the system to order r.
def CalculateMTErrorBound(dominances_sorted, r):
    result = 0
    for i in range(r, len(dominances_sorted)):
        result += dominances_sorted[i]
    return result
    
# Plot the size of the dominances of each pole and the error bound for each order of truncation.
def PlotData(dominances_sorted):
    fig, (ax_1, ax_2) = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
    
    ax_1.plot(dominances_sorted, 'bs')
    ax_1.set_title('Size of the dominance of each pole', fontsize='medium')
    ax_1.axis([0, len(dominances_sorted) - 1, 0, int(dominances_sorted[0]) + 1])
    ax_1.set_xlabel('Pole index, i')
    ax_1.set_ylabel('Dominance')
    print('Dominance')
    print(dominances_sorted)
    
    errorbounds = []
    for i in range(len(dominances_sorted)):
        errorbounds.append(CalculateMTErrorBound(dominances_sorted, i))
        
    ax_2.plot(errorbounds, 'rs')
    ax_2.set_title('Size of the MT error bound', fontsize='medium')
    ax_2.axis([0, len(errorbounds) - 1, 0, int(errorbounds[0]) + 1])
    ax_2.set_xlabel('order of reduced model')
    ax_2.set_ylabel('Error Bound')
    print('Error bounds')
    print(errorbounds)
    
    fig.suptitle('Modal Truncation', fontsize='medium')
    plt.show()
  

# Applies Modal Truncation to the Transfer Function of a state-space model
# and returns a reduced order transfer function numerator and denominator
def ApplyModalTruncation(reducedOrder, A=None, B=None, C=None, D=None, filepath=None):
    r, m, p, n, A, B, C, D = AssignVars(reducedOrder, A, B, C, D, filepath)
        
    poles, residues, leftEigVec, rightEigVec = CalculatePoleResidue(A, B, C)
    dominances = CalculateDominance(poles, residues)
    
    poles_sorted, residues_sorted, dominances_sorted, leftEigVec_sorted, rightEigVec_sorted = SortForDominance(poles, residues, dominances, leftEigVec, rightEigVec)
    
    PlotData(dominances_sorted)
    
    leftEigVec_reduced, rightEigVec_reduced = ReduceEigenVectorMatrices(leftEigVec_sorted, rightEigVec_sorted, n, r)
    residues_reduced, poles_reduced, D = ReduceTransferFunction(poles_sorted, residues_sorted, D, r)
    
    A_reduced, B_reduced, C_reduced = ReduceSystem(A, B, C, leftEigVec_reduced, rightEigVec_reduced)
    
    return A_reduced, B_reduced, C_reduced, residues_reduced, poles_reduced, D