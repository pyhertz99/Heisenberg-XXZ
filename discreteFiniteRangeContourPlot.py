"""
Creates contour plot for discrete finite
range topology
"""

import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt
from matplotlib import rcParams
from heisenbergXXZ import *
import connectivityMatrices as cm
import lowMagnetization as lm

# PARAMETERS

N = 100 #number of cells
L = 1 #number of excited cells (L <= N)
K = 1
M = int(comb(N,L))

J_xy = 1.0 #coupling constant J_x = J_y
J_z = 1.0 #coupling constant J_z

B_0 = 0.0 #magnetic field mean value
delta_B = 0.0 #magnetic field spread

t_max = 20
t_steps = 100


# COMPUTATION

#create random magnetic field array
B = np.random.uniform(low=B_0-delta_B,high=B_0+delta_B,size=N)
#B = np.load("data/B-plot-contourPlot-N100.npy")

#create connectivity matrices
M_xy, M_z = cm.discreteFiniteRange(N, K, J_xy, J_z)

#high magnetization
#PI, spin_indices = subspaceTransformationMatrix(N, L, M)
#H = createHamiltonian(N, PI, M_xy, M_z, B)

#low magnetization
H = lm.createHamiltonian(N, L, M_xy, M_z, B)

#diagonalize
spectrum, eigvecs = diagonalizeHamiltonian(H)
eigvecs_herm = eigvecs.transpose().conjugate()

#create Neel state
psi_0 = singleSpin(N,(N+1)//2)

#evolve state
psi_array = evolveState(t_max,t_steps,spectrum,eigvecs,eigvecs_herm,psi_0,M)

#high magnetization
#img = contourImg(psi_array, t_steps, PI, N)

#low magnetization
spin_basis = subspaceBasisBinary(N, L)
img = lm.contourImg(psi_array, N, M, spin_basis, t_steps)

#%% ADJUST COLORSCALE

alpha = 5

def scaleColor(x):
    return x**alpha

img_scaled = scaleColor(img)

#%% PLOTTING

fig, ax = plt.subplots(1,1, figsize=(1.5*6,2*4), dpi=300)
ax.imshow(img_scaled, aspect="auto", interpolation="none", extent=[80,120,0,t_max])
ax.set_xticks([])
ax.set_ylabel(r'$t$')
fig.tight_layout()
plt.show()
