"""
Evolves single-spin state and draws its contour plot.
"""

import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt
from matplotlib import rcParams
from heisenbergXXZ import *
import lowMagnetization as lm
import connectivityMatrices as cm

# PARAMETERS

N = 100 #number of cells
L = 1 #number of excited cells (L <= N)
K = 50 #degree of connectivity
M = int(comb(N,L))

J_xy = 1.0 #coupling constant J_x = J_y
J_z = 1.0 #coupling constant J_z

B_0 = 0.0 #magnetic field mean value
delta_B = 100.0 #magnetic field spread

t_max = 10
t_steps = 100


# COMPUTATION

#create random magnetic field array
B = np.random.uniform(low=B_0-delta_B,high=B_0+delta_B,size=N)

#create connectivity matrices
M_xy, M_z = cm.discreteFiniteRange(N, K, J_xy, J_z)

#create transformation matrix
#PI, spin_indices = subspaceTransformationMatrix(N, L, M)
spin_basis = subspaceBasisBinary(N, L)

#create hamiltonian
#H = createHamiltonian(N, PI, M_xy, M_z, B)
H = lm.createHamiltonian(N, L, M_xy, M_z, B)

#diagonalize
spectrum, eigvecs = diagonalizeHamiltonian(H)
eigvecs_herm = eigvecs.transpose().conjugate()

#create single spin state
psi_0 = singleSpin(N,(N+1)//2)
#psi_0 = randomState(M)

#evolve state
psi_array = evolveState(t_max,t_steps,spectrum,eigvecs,eigvecs_herm,psi_0,M)

#generate contour plot
#ss_img = contourImg(psi_array, t_steps, PI, N)
ss_img = lm.contourImg(psi_array, N, M, spin_basis, t_steps)

#%% ADJUST COLORSCALE

alpha = 5

def scaleColor(x):
    return x**alpha

ss_img_scaled = scaleColor(ss_img)

#%% PLOTTING

rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 30})

fig, ax = plt.subplots(1,1, figsize=(2*6,2*4), dpi=300)
ax.imshow(ss_img_scaled, aspect="auto", interpolation="none", extent=[80,120,0,t_max])
ax.set_xticks([])
ax.set_ylabel(r'$t$')
plt.show()
