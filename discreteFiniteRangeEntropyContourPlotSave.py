"""
Creates entropy contour plot for discrete finite
range topology and save.
"""

import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt
from matplotlib import rcParams
from heisenbergXXZ import *
import lowMagnetization as lm
import connectivityMatrices as cm


# PARAMETERS

N = 11 #number of cells
L = 1 #number of excited cells (L <= N)
K = 1 #degree of connectivity
M = int(comb(N,L))

J_xy = 1.0 #coupling constant J_x = J_y
J_z = 1.0 #coupling constant J_z

B_0 = 0.0 #magnetic field mean value
delta_B = 0.0 #magnetic field spread

t_max = 10
t_steps = 100


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


entropy_arrays = np.zeros((t_steps,N))
for site in range(1,N+1):
    print(site)
    ars = lm.mutualInformationArray(psi_array,np.array([6]),np.array([site]),spin_basis,M,t_steps)
    entropy_arrays[:,site-1] = ars[0]


np.save("./data/discFiniteRange-XYZ1-entropyContourPlot/N" + str(N),entropy_arrays)