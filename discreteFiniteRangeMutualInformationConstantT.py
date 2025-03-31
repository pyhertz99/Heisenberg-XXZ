"""
Compute mutual information for all sites
for constant time with discrete finite
range topology.
"""

import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt
from matplotlib import rcParams
from heisenbergXXZ import *
import lowMagnetization as lm
import connectivityMatrices as cm


# PARAMETERS

N = 10 #number of cells
L = 1 #number of excited cells (L <= N)
K = 1 #degree of connectivity
M = int(comb(N,L))

J_xy = 1.0 #coupling constant J_x = J_y
J_z = 1.0 #coupling constant J_z

B_0 = 0.0 #magnetic field mean value
delta_B = 0.5 #magnetic field spread

t_max = 0

site1 = 3

SAMPLES = 1

n = 2 #energy level (or choose different state)

# COMPUTATION

#create connectivity matrices
M_xy, M_z = cm.discreteFiniteRange(N, K, J_xy, J_z)

#create single spin state
#psi_0 = singleSpin(N,(N+1)//2)

spin_basis = subspaceBasisBinary(N, L)

entropy_arrays = np.zeros((SAMPLES,N),dtype="float")
mutual_arrays = np.zeros((SAMPLES,N),dtype="float")
for j in range(SAMPLES):
    print(j)
    
    #create random magnetic field array
    B = np.random.uniform(low=B_0-delta_B,high=B_0+delta_B,size=N)
    
    #create hamiltonian
    H = lm.createHamiltonian(N, L, M_xy, M_z, B)
    
    #diagonalize
    spectrum, eigvecs = diagonalizeHamiltonian(H)
    eigvecs_herm = eigvecs.transpose().conjugate()
    
    psi_0 = eigvecs[n]
    
    #evolve state
    psi_array = evolveState(t_max,1,spectrum,eigvecs,eigvecs_herm,psi_0,M)
    
  
    for i in range(1,N):
        print(i)
        sites1 = np.array([site1])
        sites2 = np.array([i])
        
        ars = lm.mutualInformationArray(psi_array,sites1,sites2,spin_basis,M,1)
        entropy_arrays[j,i-1] = ars[0][0]
        mutual_arrays[j,i-1] = ars[1][0]

#%% AVERAGE

entropy_means = np.zeros(N,dtype="float")
mutual_means = np.zeros(N,dtype="float")
entropy_sigma= np.zeros(N,dtype="float")
mutual_sigma = np.zeros(N,dtype="float")
for i in range(N):
    entropy_means[i] = np.mean(entropy_arrays[:,i])
    mutual_means[i] = np.mean(mutual_arrays[:,i])
    entropy_sigma[i] = np.std(entropy_arrays[:,i])
    mutual_sigma[i] = np.std(mutual_arrays[:,i])

#%% PLOTTING

rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 30})

xs = np.arange(1,N+1)
#xs = np.delete(xs,site1-1)
#entropy_means_plot = np.delete(entropy_means,site1-1)
#mutual_means_plot = np.delete(mutual_means,site1-1)

fig, ax = plt.subplots(1,1, figsize=(2*6,2*4),dpi=300)
#plt.plot(xs,entropy_means,color='r',label=r"$S(B)$",marker="o")
plt.plot(xs,mutual_means,color='b',label=r"$I(A:B)$",marker="o")
#plt.plot(ts, mutual_array_ss,color='r',label=r"$I(A:B)$")
ax.set_xlabel(r'$x$')
#ax.set_ylabel(r'$S$')
#ax.set_ylim(0,1)
ax.legend()

plt.show()
