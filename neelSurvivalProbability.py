"""
Evolves Neel state with given parameters
and plots its survival probability as a function of time.
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
J_z = 0.0 #coupling constant J_z

B_0 = 0.0 #magnetic field mean value
delta_B = 0.0 #magnetic field spread

t_max = 6.6
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

#create Neel state
psi_0 = neel(N, M, spin_indices)

#evolve state
psi_array = evolveState(t_max,t_steps,spectrum,eigvecs,eigvecs_herm,psi_0,M)

#compute survival probability
neel_survival_array = survivalProbability(psi_array, t_steps)
ts = np.linspace(0,t_max,t_steps)

#%% PLOTTING

rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 30})

fig, ax = plt.subplots(1,1, figsize=(2*6,2*4), dpi=300)
plt.plot(ts, neel_survival_array)
ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$\left< P \right>$')
ax.set_ylim(0,1)

plt.show()
