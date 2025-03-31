"""
Compute spectral characteristics of
system with 2D grid topology..
"""

import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt
from matplotlib import rcParams
from heisenbergXXZ import *
from quantumChaos import *
import connectivityMatrices as cm


# SYSTEM PARAMETERS

n = 5 #sites in x axis
m = 2 #sites in y axis

N = m*n #number of sites
L = N//2 #number of excited sites (L <= N)
M = int(comb(N,L))

J_xy = 0.2 #coupling constant J_x = J_y
J_z = 0.3 #coupling constant J_z

B_0 = 0.0 #magnetic field mean value
delta_B = 0.0 #magnetic field spread


# COMPUTE SPECTRUM

#create random magnetic field array
B = np.random.uniform(low=B_0-delta_B,high=B_0+delta_B,size=N)

#create connectivity matrices
M_xy, M_z = cm.grid2D(m, n, J_xy, J_z)

#create transformation matrix
PI, spin_indices = subspaceTransformationMatrix(N, L, M)

#create hamiltonian
H = createHamiltonian(N, PI, M_xy, M_z, B)

#diagonalize
spectrum, eigvecs = diagonalizeHamiltonian(H)

#%% ANALYZE SPECTRUM

DEG = 7 #degree of polynom used for unfolding
BINS = 20 #number of bins in histogram

#number of energy levels to cut from each edge
#(could help when fit diverges)
CUT_EDGES = 10

points = spectrum.size

#compute unfolded spectrum
unfoldedSpectrum, polynom = unfoldSpectrum(spectrum, DEG, CUT_EDGES)

#compute histogram and fit with Brody distribution
level_dif = levelDifference(unfoldedSpectrum)
hist, bin_centres, params, cov = brodyFit(level_dif, BINS)
print(f"Fit with Brody distribution:\n")
print(f"beta       = {params[0]}")
print(f"sigma_beta = {np.sqrt(cov[0,0])}\n\n")

#compute mean ratio of consecutive levels
r, sigma_r = ratioConsecutiveLevels(spectrum)
print(f"Statistical test:\n")
print(f"mean_r  = {r}")
print(f"sigma_r = {sigma_r}")

#%% PLOT INTEGRATED SPECTRUM

rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 30})

fig, ax1 = plt.subplots(1,1,figsize=(2*6,2*4))

int_density = integratedLevelDensity(spectrum)

xs = np.linspace(spectrum[0],spectrum[-1],1000)
plt.scatter(spectrum,int_density,alpha=0.3, color="orange",label="spectrum")
plt.plot(xs, polynom(xs),color="k",label="polynomial fit")
plt.legend()

print(f"Cuts at: {spectrum[CUT_EDGES]}")
print(f"          {spectrum[points-CUT_EDGES]}")

plt.show()

#%% PLOT HISTOGRAM

rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 30})

fig, ax1 = plt.subplots(1,1,figsize=(2*6,2*4))

xs = np.linspace(bin_centres[0],bin_centres[-1],1000)
plt.step(bin_centres,hist,where='mid',linestyle='-',color="b")
plt.plot(xs,brodyDistribution(xs, params[0], params[1]),color="k",label="Brody fit")
plt.legend()

plt.show()