"""
Compute spectral characteristics of
system with 2D grid topology with respect
to B.
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
L = N//2 #number of excited cells (L <= N)
M = int(comb(N,L))

J_xy = 1.0 #coupling constant J_x = J_y
J_z = 1.0

SAMPLES = 3 #number of samples for each parameter setting
POINTS = 40 #number of dB values
dB_min = 0.5
dB_max = 10
dB_array = np.linspace(dB_min,dB_max,POINTS)

B_0 = 0.0 #magnetic field mean value

# COMPUTE SPECTRUM

spectra = np.zeros((POINTS,SAMPLES,M))
for i in range(POINTS):
    
    delta_B = dB_array[i]
    print(delta_B)
    for j in range(SAMPLES):
        
        #create random magnetic field array
        B = np.random.uniform(low=B_0-delta_B,high=B_0+delta_B,size=N)
        
        #create connectivity matrices
        M_xy, M_z = cm.grid2D(m, n, J_xy, J_z, twist_n=True, twist_m=False)
        
        #create transformation matrix
        PI, spin_indices = subspaceTransformationMatrix(N, L, M)
        
        #create hamiltonian
        H = createHamiltonian(N, PI, M_xy, M_z, B)
        
        #diagonalize
        spectrum, eigvecs = diagonalizeHamiltonian(H)
        
        spectra[i,j] = spectrum

#%% ANALYZE SPECTRUM

DEG = 7 #degree of polynom used for unfolding
BINS = 10 #number of bins in histogram

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

r_0 = 2*np.log(2) - 1
r_1 = 5 - 2*np.sqrt(5)

alphas = (rs - r_0)/(r_1-r_0)

betas_mean = np.zeros(POINTS)
betas_sigma = np.zeros(POINTS)
alphas_mean = np.zeros(POINTS)
alphas_sigma = np.zeros(POINTS)
for i in range(POINTS):
    betas_mean[i] = np.mean(betas[i])
    betas_sigma[i] = np.std(betas[i])
    alphas_mean[i] = np.mean(alphas[i])
    alphas_sigma[i] = np.std(alphas[i])

#%% PLOT HISTOGRAM

rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 30})

fig, ax1 = plt.subplots(1,1,figsize=(2*6,2*4))

# Plotting the first dataset on the left y-axis
ax1.errorbar(dB_array, alphas_mean, yerr=alphas_sigma,marker="o", label=r"$\alpha$", color="blue")
ax1.set_xlabel(r"$\Delta B$")
#ax1.tick_params(axis='y', labelcolor="blue")
#ax1.legend(loc="upper right")

# Creating a second y-axis

# Plotting the second dataset on the right y-axis
ax1.errorbar(dB_array, betas_mean, yerr=betas_sigma,marker="o", label=r"$\beta$", color="red")
#ax1.tick_params(axis='y', labelcolor="red")
#ax2.set_yticks(np.linspace(-1,1,5))
#ax2.legend(loc="upper right")
ax1.set_yticks(np.linspace(-0.25,1.25,7))
ax1.set_xticks(np.linspace(1,10,7))

fig.legend(bbox_to_anchor=(0.9, 0.88))

plt.show()