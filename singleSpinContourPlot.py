"""
Evolves single-spin state (in the middle)
with given parameters and draws its contour plot.
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
J_z = 10.0 #coupling constant J_z

B_0 = 0.0 #magnetic field mean value
delta_B = 0.0 #magnetic field spread

t_max = 10
t_steps = 100


# COMPUTATION

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

#generate contour plot
ss_img = contourImg(psi_array, t_steps, PI, N)


#%% PLOTTING

rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 30})

fig, ax = plt.subplots(1,1, figsize=(2*6,2*4), dpi=300)
ax.imshow(ss_img, aspect="auto", interpolation="none", extent=[80,120,0,t_max])
ax.set_xticks([])
ax.set_ylabel(r'$t$')
plt.show()
