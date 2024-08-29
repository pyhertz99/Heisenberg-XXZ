"""
Creates level dynamics plot.
"""

import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt
from matplotlib import rcParams
from heisenbergXXZ import *


# PARAMETERS

N = 10 #number of cells
L = N//2 #number of excited cells (L <= N)
M = int(comb(N,L))

J_xy = 1.0 #coupling constant J_x = J_y

B_0 = 0.0 #magnetic field mean value
delta_B = 0.0 #magnetic field spread

B = np.random.uniform(low=B_0-delta_B,high=B_0+delta_B,size=N)

# COMPUTATION

J_zs = np.linspace(-2.5, 2.5, 11)
lines = np.zeros((len(J_zs), M))
neel_line = np.zeros(len(J_zs))
dw_line = np.zeros(len(J_zs))

for (i, J_z) in enumerate(J_zs):
    M_xy, M_z = createConnectivityMatrices(N, J_xy, J_z)
    PI, spin_indices = subspaceTransformationMatrix(N, L, M)
    H = createHamiltonian(N, PI, M_xy, M_z, B)
      
    spectrum, eigvecs = diagonalizeHamiltonian(H)

    dw_state = domainWall(M)
    neel_state = neel(N, M, spin_indices)

    dw_line[i] = dw_state.transpose() @ H @ dw_state
    neel_line[i] = neel_state.transpose() @ H @ neel_state

    lines[i] = spectrum

#%% PLOTTING

rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 30})

fig, ax = plt.subplots(1,1, figsize=(2*6,2*4), dpi=300)
ax.plot(J_zs, lines, color="black", alpha=0.4)
ax.plot(J_zs, dw_line, color="red", alpha=0.6)
ax.plot(J_zs, neel_line, color="blue", alpha=0.4)
ax.set_xlabel(r"$J_z$")
ax.set_ylabel(r"$E$")

plt.show()