"""
Save spectra for 
discrete finite range topology for analysis
with respect to K.
"""

import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt
from matplotlib import rcParams
from heisenbergXXZ import *
import connectivityMatrices as cm


# PARAMETERS

N = 10 #number of cells
L = N//2 #number of excited cells (L <= N)
M = int(comb(N,L))

J_xy = 1.0 #coupling constant J_x = J_y
J_z = 1.0

B_0 = 0.0 #magnetic field mean value
delta_Bs = np.array([0.1,0.2,0.5,1.0,1.5,2.0])
nB = delta_Bs.size

samples = 20

spectra = np.zeros((nB,N//2,samples,M),dtype="float")

# COMPUTATION
for i in range(nB):
    delta_B = delta_Bs[i]
    print(delta_B)
    for K in range(1,N//2 + 1):
        print(K)
        
        for sample in range(samples):
            print("sample " + str(sample+1) + "/" + str(samples))
            
            #create random magnetic field array
            B = np.random.uniform(low=B_0-delta_B,high=B_0+delta_B,size=N)
            
            #create connectivity matrices
            M_xy, M_z = cm.discreteFiniteRange(N, K, J_xy, J_z)
            
            #create transformation matrix
            PI, spin_indices = subspaceTransformationMatrix(N, L, M)
            
            #create hamiltonian
            H = createHamiltonian(N, PI, M_xy, M_z, B)
            
            #diagonalize
            spectrum, eigvecs = diagonalizeHamiltonian(H)
            
            spectra[i,K-1,sample] = spectrum
            
np.save("./data/discFiniteRange-XYZ1-smp20/spectra-N" + str(N),spectra)