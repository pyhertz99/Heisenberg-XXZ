"""
Compute time evolution of mutual information
in discrete finite range topology.¨
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
delta_B = 1.0 #magnetic field spread

t_max = 2
t_steps = 100

site1 = 6
site2 = 3


# COMPUTATION

#create random magnetic field array
B = np.random.uniform(low=B_0-delta_B,high=B_0+delta_B,size=N)

#create connectivity matrices
M_xy, M_z = cm.discreteFiniteRange(N, K, J_xy, J_z)

#create transformation matrix
#PI, spin_indices = subspaceTransformationMatrix(N, L, M)

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

#compute entanglement entropy
#entropy_array_ss = entanglementEntropyArray(psi_array, t_steps, PI, site, N)
spin_basis = subspaceBasisBinary(N, L)

sites1 = np.array([site1])
sites2 = np.array([site2])

ars = lm.mutualInformationArray(psi_array,sites1,sites2,spin_basis,M,t_steps)
entropy_array_ss = ars[0]
mutual_array_ss = ars[1]

ts = np.linspace(0,t_max,t_steps) 

#%% PLOTTING

rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 30})

fig, ax = plt.subplots(1,1, figsize=(2*6,2*4),dpi=300)
plt.plot(ts, entropy_array_ss,color='r',label=r"$S(B)$")
plt.plot(ts, mutual_array_ss,color='b',label=r"$I(A:B)$")
ax.set_xlabel(r'$t$')
#ax.set_ylabel(r'$S$')
#ax.set_ylim(0,1)
ax.legend()

plt.show()
