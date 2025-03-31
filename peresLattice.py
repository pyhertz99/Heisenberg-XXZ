"""
Compute peres lattice.
"""

import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt
from matplotlib import rcParams
from heisenbergXXZ import *
import connectivityMatrices as cm


N = 10 #number of cells
L = N//2 #number of excited cells (L <= N)
K = 2
M = int(comb(N,L))

J_xy = 1.0 #coupling constant J_x = J_y
J_z = 1.0

B_0 = 0.0 #magnetic field mean value
delta_B = 0.5

site = 5


#create random magnetic field array
#B = np.random.uniform(low=B_0-delta_B,high=B_0+delta_B,size=N)

#create connectivity matrices
M_xy, M_z = cm.discreteFiniteRange(N, K, J_xy, J_z)

#create transformation matrix
PI, spin_indices = subspaceTransformationMatrix(N, L, M)

#create hamiltonian
H = createHamiltonian(N, PI, M_xy, M_z, B)

#diagonalize
spectrum, eigvecs = diagonalizeHamiltonian(H)
eigvecs_herm = eigvecs.transpose().conjugate()


#create subspace operator

#sigma_z on single site
#A = createOperator(N, [sigma_z], [site-1])

#x magnetization on all sites
ops = []
for i in range(N):
    ops.append(sigma_x)
A = createOperator(N, ops, np.arange(N+1))


A_sub = PI @ A @ PI.T
lattice = peresLattice(A_sub, spectrum, eigvecs_herm)

#%%

rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 50})

fig, axs = plt.subplots(2,3,figsize=(4.5*6,3.9*4),layout="tight",dpi=200)

axs[0,0].scatter(lattice[:,0],lattice[:,1])
axs[0,0].scatter([],[], label=r"$K=2$",marker="")
# axs[0,1].scatter(lattice3[:,0],lattice3[:,1])
# axs[0,1].scatter([],[], label=r"$K=6$",marker="")
# axs[0,2].scatter(lattice4[:,0],lattice4[:,1])
# axs[0,2].scatter([],[], label=r"$K=7$",marker="")
# axs[1,0].scatter(latticeM1[:,0],latticeM1[:,1])
# axs[1,1].scatter(latticeM3[:,0],latticeM3[:,1])
# axs[1,2].scatter(latticeM4[:,0],latticeM4[:,1])


axs[0,0].legend(prop={'size': 30})
axs[0,1].legend(prop={'size': 30})
axs[0,2].legend(prop={'size': 30})

axs[0,0].set_ylim(-1.1,1.1)
axs[0,1].set_ylim(-1.1,1.1)
axs[0,2].set_ylim(-1.1,1.1)
axs[1,0].set_ylim(-1.1,1.1)
axs[1,1].set_ylim(-1.1,1.1)
axs[1,2].set_ylim(-1.1,1.1)

axs[1,0].set_xlabel(r"$E_n$")
axs[1,1].set_xlabel(r"$E_n$")
axs[1,2].set_xlabel(r"$E_n$")

axs[0,0].set_ylabel(r"$\left< \hat{\sigma}^{(j)}_z \right>_n$")
axs[1,0].set_ylabel(r"$\left< \hat{M}_x \right>_n$")

axs[0,0].set_xlim(-30,30)
axs[1,0].set_xlim(-30,30)

axs[0,1].set_xlim(-40,80)
axs[1,1].set_xlim(-40,80)

axs[0,2].set_xlim(-30,100)
axs[1,2].set_xlim(-30,100)


axs[0,0].set_xticklabels([])
axs[0,1].set_xticklabels([])
axs[0,2].set_xticklabels([])


axs[0,1].set_yticklabels([])
axs[0,2].set_yticklabels([])

axs[1,1].set_yticklabels([])
axs[1,2].set_yticklabels([])

axs[0,0].set_yticks([-1,-0.5,0,0.5,1])
axs[1,0].set_yticks([-1,-0.5,0,0.5,1])

axs[1,0].set_xticks([-30,-15,0,15,30])
axs[1,1].set_xticks([-40,0,40,80])
axs[1,2].set_xticks([-30,0,30,60,90])


ax1.set_xlabel(r"$E_n$")
ax1.set_ylabel(r"$\left< \hat{M}_x \right>_n$")

#ax1.tick_params(axis='y', labelcolor="blue")
#ax1.legend(loc="upper right")

#ax1.legend(loc="upper right",prop={'size': 20})
#ax1.set_yticks(np.linspace(-0.4,0.4,5))
#ax1.set_xticks(np.linspace(-60.0,100.0,5))

plt.show()
