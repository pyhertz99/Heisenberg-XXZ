"""
Computes average mutual information in time
for discrete finite range topology.
"""

import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt
from matplotlib import rcParams
from heisenbergXXZ import *
import lowMagnetization as lm
import connectivityMatrices as cm


# PARAMETERS

N = 11 #number of cells
L = 1 #number of excited cells (L <= N)
K = 1 #degree of connectivity
M = int(comb(N,L))

J_xy = 1.0 #coupling constant J_x = J_y
J_z = 1.0 #coupling constant J_z

B_0 = 0.0 #magnetic field mean value
delta_B = 1.5 #magnetic field spread

t_max = 10
t_steps = 100

site1 = 3
site2 = 5

SAMPLES = 3 #number of samples for each parameter setting


# COMPUTATION

#create connectivity matrices
M_xy, M_z = cm.discreteFiniteRange(N, K, J_xy, J_z)

mutinf_array = np.zeros((SAMPLES,t_steps),dtype="float")
for j in range(SAMPLES):
    
    print(j)

    #create random magnetic field array
    B = np.random.uniform(low=B_0-delta_B,high=B_0+delta_B,size=N)
    
    #create hamiltonian
    #H = createHamiltonian(N, PI, M_xy, M_z, B)
    H = lm.createHamiltonian(N, L, M_xy, M_z, B)
    
    #diagonalize
    spectrum, eigvecs = diagonalizeHamiltonian(H)
    eigvecs_herm = eigvecs.transpose().conjugate()
    
    #create single spin state
    psi_0 = singleSpin(N,(N+1)//2)
    
    #evolve state
    psi_array = evolveState(t_max,t_steps,spectrum,eigvecs,eigvecs_herm,psi_0,M)
    
    #compute mutual information
    spin_basis = subspaceBasisBinary(N, L)
    
    sites1 = np.array([site1])
    sites2 = np.array([site2])

    ars = lm.mutualInformationArray(psi_array,sites1,sites2,spin_basis,M,t_steps)
    
    mutinf_array[j] = ars[1]
    
#%% AVERAGE

mutinf_means = np.zeros(t_steps,dtype="float")
for i in range(t_steps):
    mutinf_t = mutinf_array[:,i]
    mutinf_means[i] = np.mean(mutinf_t)

#%% PLOTTING

ts = np.linspace(0,t_max,t_steps) 

rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 30})

fig, ax = plt.subplots(1,1, figsize=(2*6,2*4),dpi=300)
plt.plot(ts, mutinf_means)
ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$I(A:B)$')
#ax.set_ylim(-0.005,0.1)
ax.legend(prop={'size': 25})

plt.show()


#%%


ts = np.linspace(0,t_max,t_steps) 

rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 30})

fig, ax = plt.subplots(1,1, figsize=(2*6,2*4),dpi=300)
plt.plot(ts, mutinf_00,label=r"$\Delta B = 0.0$")
plt.plot(ts, mutinf_10,label=r"$\Delta B = 1.0$")
plt.plot(ts, mutinf_15,label=r"$\Delta B = 1.5$")
plt.plot(ts, mutinf_20,label=r"$\Delta B = 2.0$")
ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$I(A:B)$')
#ax.set_ylim(-0.005,0.1)
ax.legend(prop={'size': 25})

plt.show()
