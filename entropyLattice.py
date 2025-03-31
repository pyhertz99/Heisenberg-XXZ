"""
Computes entropy lattice.
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
K = 4
M = int(comb(N,L))

J_xy = 1.0 #coupling constant J_x = J_y
J_z = 1.0 #coupling constant J_z

B_0 = 0.0 #magnetic field mean value
delta_B = 2.2 #magnetic field spread

t_max = 100


# COMPUTATION

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
eigvecs_herm = eigvecs.transpose().conjugate()

#create Neel state
#psi_0 = neel(N, M, spin_indices)

#evolve state
#psi = evolveState(t_max,1,spectrum,eigvecs,eigvecs_herm,psi_0,M)[0]
psi = eigvecs_herm[0]


#compute entanglement entropy
rho = densityMatrix(psi,PI)
entropyLattice = entropyLattice(rho,N)

xs, ys = latticePoints(N)
zs = np.array([],dtype=float)

for i in range(N):
    zs = np.concatenate((zs,entropyLattice[i]))


rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 30})


fig, ax = plt.subplots(1,1, figsize=(2*6,2*4))
plt.scatter(xs, ys, c=zs, s=1600, vmax=2.0)
ax.set_xticks([])
ax.set_yticks([])
plt.colorbar()

ax.set_ylim(-1,9)
ax.set_xlim(-5.7,5.7)

plt.show()