"""
Compute information lattice.
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
delta_B = 0.5 #magnetic field spread

t_max = 100.0


# COMPUTATION

#create random magnetic field array
B = np.random.uniform(low=B_0-delta_B,high=B_0+delta_B,size=N)

#create connectivity matrices
M_xy, M_z = cm.chain(N, J_xy, J_z,boundary=0.0)

#create transformation matrix
PI, spin_indices = subspaceTransformationMatrix(N, L, M)

#create hamiltonian
H = createHamiltonian(N, PI, M_xy, M_z, B)

#diagonalize
spectrum, eigvecs = diagonalizeHamiltonian(H)
eigvecs_herm = eigvecs.transpose().conjugate()

#create Neel state
psi_0 = domainWall(M)

#evolve state
psi = evolveState(t_max,1,spectrum,eigvecs,eigvecs_herm,psi_0,M)[0]
#psi = eigvecs_herm[0]

#compute entanglement entropy
rho = densityMatrix(psi,PI)
informationLattice = informationLattice(rho,N)

xs, ys = latticePoints(N)
zs3 = np.array([],dtype=float)

for i in range(N):
    zs3 = np.concatenate((zs3,informationLattice[i]))

#%%

rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 55})


fig, axs = plt.subplots(1,4, figsize=(6*6,2.2*4),layout="constrained",dpi=200)
axs[0].scatter(xs, ys, c=zs, s=2200, vmax=1.0, vmin=0.0)
axs[1].scatter(xs, ys, c=zs1, s=2200, vmax=1.0, vmin=0.0)
axs[2].scatter(xs, ys, c=zs2, s=2200, vmax=1.0, vmin=0.0)
plot = axs[3].scatter(xs, ys, c=zs3, s=2200, vmax=1.0, vmin=0.0)

axs[0].axis('off')
axs[1].axis('off')
axs[2].axis('off')
axs[3].axis('off')

axs[0].set_xticks([])
axs[0].set_yticks([])
axs[1].set_xticks([])
axs[1].set_yticks([])
axs[2].set_xticks([])
axs[2].set_yticks([])
axs[3].set_xticks([])
axs[3].set_yticks([])

fig.colorbar(plot)

axs[0].set_ylim(-1,9)
axs[0].set_xlim(-5.7,5.7)

axs[1].set_ylim(-1,9)
axs[1].set_xlim(-5.7,5.7)

axs[2].set_ylim(-1,9)
axs[2].set_xlim(-5.7,5.7)

axs[3].set_ylim(-1,9)
axs[3].set_xlim(-5.7,5.7)

axs[0].set_title(r"$t=0.0$", fontsize=55)
axs[1].set_title(r"$t=0.7$",  fontsize=55)
axs[2].set_title(r"$t=1.0$",  fontsize=55)
axs[3].set_title(r"$t=100.0$",  fontsize=55)

plt.show()