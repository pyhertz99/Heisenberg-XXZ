"""
Creates samples of single-spin state evolution for given
parameters and saves it as binary files.
"""

import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt
from matplotlib import rcParams
from heisenbergXXZ import *


# PARAMETERS
N = 19 #number of cells
L = 1 #number of excited cells (L <= N)
M = int(comb(N,L))

J_xy = 1.0 #coupling constant J_x = J_y
J_z_array = [0,1,10] #coupling constant J_z

B_0 = 0.0 #magnetic field mean value
delta_B_array = [0,0.2,0.5,1,2,5] #magnetic field spread

t_max = 10
t_steps = 100

folder = "./data/"

len_delta_B_array = len(delta_B_array)
len_J_z_array = len(J_z_array)

#specify how many samples to create
samples_array = np.full((len_delta_B_array,len_J_z_array),50) #50 for dB != 0
samples_array[0] = np.full((len_J_z_array),1) #just one for dB=0

# COMPUTATION

for i in range(len(J_z_array)):
    J_z = J_z_array[i]
    J_z_string = f"{J_z:.2f}"
    print("J_z = " + J_z_string)
    
    for j in range(len(delta_B_array)):
        delta_B = delta_B_array[j]
        delta_B_string = f"{delta_B:.2f}"
        print("dB = " + delta_B_string)
        
        samples = samples_array[j,i]
        for sample in range(samples):
            print("sample " + str(sample+1) + "/" + str(samples))
            
            #create random magnetic field array
            B = np.random.uniform(low=B_0-delta_B,high=B_0+delta_B,size=N)
            
            #create connectivity matrices
            M_xy, M_z = createConnectivityMatrices(N, J_xy, J_z)
            
            #create transformation matrix
            PI, spin_indices = subspaceTransformationMatrix(N, L, M)
            
            #create hamiltonian
            H = createHamiltonian(N, PI, M_xy, M_z, B)
            
            #diagonalize
            spectrum, eigvecs = diagonalizeHamiltonian(H)
            eigvecs_herm = eigvecs.transpose().conjugate()
            
            #create single spin state
            psi_0 = singleSpin(N,(N+1)//2)
            
            #evolve state
            psi_array = evolveState(t_max,t_steps,spectrum,eigvecs,eigvecs_herm,psi_0,M)
            
            np.save(folder + "single-spin/N" + str(N) + "-Z" 
                    + J_z_string + "-dB" + delta_B_string
                    + "-tMax" + str(t_max) + "-tSteps" 
                    + str(t_steps) + "-" + str(sample)
                    + ".npy", psi_array)
                
