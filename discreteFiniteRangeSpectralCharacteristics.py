"""
Compute spectral characteristics of
system with discrete finite range
topology.
"""

import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt
from matplotlib import rcParams
from heisenbergXXZ import *
from quantumChaos import *
import connectivityMatrices as cm


# SYSTEM PARAMETERS

N = 14 #number of cells
K = 5
L = N//2 #number of excited cells (L <= N)
M = int(comb(N,L))

J_xy = 1.0 #coupling constant J_x = J_y
J_z = 1.0 #coupling constant J_z

B_0 = 0.0 #magnetic field mean value
delta_B = 0.2 #magnetic field spread


# COMPUTE SPECTRUM

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

#%% ANALYZE SPECTRUM

DEG = 7 #degree of polynom used for unfolding
BINS = 180 #number of bins in histogram

#number of energy levels to cut from each edge
#(could help when fit diverges)
CUT_EDGES = 10

points = spectrum.size

#compute unfolded spectrum
unfoldedSpectrum, polynom = unfoldSpectrum(spectrum, DEG, CUT_EDGES)

#compute histogram and fit with Brody distribution
level_dif = levelDifference(unfoldedSpectrum)
hist2, bin_centres2, params2, cov2 = brodyFit(level_dif, BINS)
print(f"Fit with Brody distribution:\n")
print(f"beta       = {params2[0]}")
print(f"sigma_beta = {np.sqrt(cov2[0,0])}\n\n")

#compute mean ratio of consecutive levels
r2, sigma_r2 = ratioConsecutiveLevels(spectrum)
print(f"Statistical test:\n")
print(f"mean_r  = {r2}")
print(f"sigma_r = {sigma_r2}")

#%% PLOT HISTOGRAM
 
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 40})

fig, axs = plt.subplots(1,2,figsize=(3*6,2*4))

xs = np.linspace(0,bin_centres[-1],1000)
xs2 = np.linspace(0,bin_centres2[-1],1000)

axs[0].step(bin_centres,hist,where='mid',linestyle='-',color="b")
axs[0].plot(xs,brodyDistribution(xs, params[0]),color="r")
axs[1].step(bin_centres2,hist2,where='mid',linestyle='-',color="b",label=r"histogram")
axs[1].plot(xs2,brodyDistribution(xs2, params2[0]),color="r",label=r"fit Brody")

plt.rcParams['axes.titley'] = -0.1    # y is in axes-relative coordinates.
plt.rcParams['axes.titlepad'] = 0
axs[1].set_xlim(0,xs[-1])
axs[0].set_ylim(0,2)
axs[1].set_ylim(0,2)

axs[1].yaxis.set_ticklabels([])

axs[1].legend(prop={'size': 30})
#axs[0].legend(prop={'size': 30})

plt.tight_layout()

plt.show()