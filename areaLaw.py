"""
Computes mutual information between connected
sites region and its complement.
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
J_z = 1.0 #coupling constant J_z

B_0 = 0.0 #magnetic field mean value
delta_B = 0.2 #magnetic field spread

n = 0 #eigenstate level


# COMPUTATION

#create random magnetic field array
B = np.random.uniform(low=B_0-delta_B,high=B_0+delta_B,size=N)

#create connectivity matrices
M_xy, M_z = cm.chain(N, J_xy, J_z)

#create transformation matrix
PI, spin_indices = subspaceTransformationMatrix(N, L, M)

#create hamiltonian
H = createHamiltonian(N, PI, M_xy, M_z, B)

#diagonalize
spectrum, eigvecs = diagonalizeHamiltonian(H)
eigvecs_herm = eigvecs.transpose().conjugate()


#psi_0 = domainWall(M)
#psi = evolveState(t_max,1,spectrum,eigvecs,eigvecs_herm,psi_0,M)[0]

psi = eigvecs_herm[n]

#compute entanglement entropy
rho = densityMatrix(psi,PI)
areaLaw = areaLaw(rho,N)

#%%
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 50})


fig, ax = plt.subplots(1,1, figsize=(2*6,2*4))
plt.plot(np.arange(1,N),areaLaw,marker="o",label=r"$n=0$",markersize=15)
plt.legend(prop={'size': 25})
#ax.set_xticks([2,4,6,8])
#ax.set_yticks([])
#plt.colorbar()

#ax.set_xlabel(r"$\Delta B$")
ax.set_ylabel(r"$I$")
ax.set_xlabel(r"sites")

ax.set_ylim(0,6)
#ax.set_xlim(-5.7,5.7)

plt.show()