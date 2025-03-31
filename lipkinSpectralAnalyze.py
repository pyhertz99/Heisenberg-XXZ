"""
Compute spectral characteristics of
system with lipkin topology with respect
to J_z.
"""

import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt
from matplotlib import rcParams
from heisenbergXXZ import *
from quantumChaos import *
import connectivityMatrices as cm


# SYSTEM PARAMETERS

N = 10 #number of cells
L = N//2 #number of excited cells (L <= N)
M = int(comb(N,L))

J_xy = 1.0 #coupling constant J_x = J_y

SAMPLES = 5 #number of samples for each parameter setting
POINTS = 20 #number of J_z values
J_z_min = 0
J_z_max = 10
J_z_array = np.linspace(J_z_min,J_z_max,POINTS) #coupling constant J_z

B_0 = 0.0 #magnetic field mean value
delta_B = 1.0 #magnetic field spread


# COMPUTE SPECTRUM

spectra = np.zeros((POINTS,SAMPLES,M))
for i in range(POINTS):
    
    J_z = J_z_array[i]
    print(J_z)
    for j in range(SAMPLES):
        
        #create random magnetic field array
        B = np.random.uniform(low=B_0-delta_B,high=B_0+delta_B,size=N)
        
        #create connectivity matrices
        M_xy, M_z = cm.lipkin(N, J_xy, J_z)
        
        #create transformation matrix
        PI, spin_indices = subspaceTransformationMatrix(N, L, M)
        
        #create hamiltonian
        H = createHamiltonian(N, PI, M_xy, M_z, B)
        
        #diagonalize
        spectrum, eigvecs = diagonalizeHamiltonian(H)
        
        spectra[i,j] = spectrum

#%% ANALYZE SPECTRUM

DEG = 7 #degree of polynom used for unfolding
BINS = 20 #number of bins in histogram

#number of energy levels to cut from each edge
#(could help when fit diverges)
CUT_EDGES = 0

betas = np.zeros((POINTS,SAMPLES))
rs = np.zeros((POINTS,SAMPLES))
for i in range(POINTS):
    for j in range(SAMPLES):
        spectrum = spectra[i,j]
        
        #compute Brody fit
        unfoldedSpectrum, polynom = unfoldSpectrum(spectrum, DEG, CUT_EDGES)
        level_dif = levelDifference(unfoldedSpectrum)
        hist, bin_centres, params, cov = brodyFit(level_dif, BINS)
        betas[i,j] = params[0]
        
        #compute mean ratio of consecutive levels
        r, sigma_r = ratioConsecutiveLevels(spectrum)
        rs[i,j] = r

betas_mean = np.zeros(POINTS)
betas_sigma = np.zeros(POINTS)
rs_mean = np.zeros(POINTS)
rs_sigma = np.zeros(POINTS)
for i in range(POINTS):
    betas_mean[i] = np.mean(betas[i])
    betas_sigma[i] = np.std(betas[i])
    rs_mean[i] = np.mean(rs[i])
    rs_sigma[i] = np.std(rs[i])

#%% PLOT HISTOGRAM

rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 30})

fig, ax1 = plt.subplots(1,1,figsize=(2*6,2*4))

# Plotting the first dataset on the left y-axis
ax1.errorbar(J_z_array, rs_mean, yerr=rs_sigma, marker="o", label=r"$\langle r \rangle$", color="blue")
ax1.set_xlabel(r"$J_z$")
ax1.tick_params(axis='y', labelcolor="blue")
ax1.set_yticks(np.linspace(0,0.4,5))
#ax1.legend(loc="upper right")

# Creating a second y-axis
ax2 = ax1.twinx()

# Plotting the second dataset on the right y-axis
ax2.errorbar(J_z_array, betas_mean, yerr=betas_sigma, marker="o", label=r"$\beta$", color="red")
ax2.tick_params(axis='y', labelcolor="red")
ax2.set_yticks(np.linspace(-1,1,5))
#ax2.legend(loc="upper right")

fig.legend(bbox_to_anchor=(0.9, 0.68))

plt.show()