"""
Analyze spectra of discrete finite range
topology with respect to B.
"""

import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt
from matplotlib import rcParams
from heisenbergXXZ import *
import connectivityMatrices as cm
from quantumChaos import *


# PARAMETERS

N = 10 #number of cells
L = N//2 #number of excited cells (L <= N)
K = 1
M = int(comb(N,L))

J_xy = 1.0 #coupling constant J_x = J_y
J_z = 1.0

B_0 = 0.0 #magnetic field mean value
delta_B_array = np.linspace(0,1,3) #magnetic field spread

samples = 3

spectra = np.zeros((delta_B_array.size,samples,M),dtype="float")

# COMPUTATION

for j in range(len(delta_B_array)):
    delta_B = delta_B_array[j]
    delta_B_string = f"{delta_B:.2f}"
    print("dB = " + delta_B_string)

    for sample in range(samples):
        print("sample " + str(sample+1) + "/" + str(samples))
        
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
        
        spectra[j,sample] = spectrum
#%%

POINTS = len(delta_B_array)
SAMPLES = samples


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
    
#%%

rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 30})

fig, ax1 = plt.subplots(1,1,figsize=(2*6,2*4))

# Plotting the first dataset on the left y-axis
ax1.plot(delta_B_array, alphas_mean,marker="o", label=r"$K = 1$")
ax1.plot(delta_B_array, betas_mean,marker="o", label=r"$K = 1$")

# ax1.plot(delta_B_array, alphas_mean_3[1:],marker="o", label=r"$K = 3$")
# ax1.plot(delta_B_array, alphas_mean_5[1:],marker="o", label=r"$K = 5$")
# ax1.plot(delta_B_array, alphas_mean_7[1:],marker="o", label=r"$K = 7$")
# ax1.plot(delta_B_array, alphas_mean_9[1:],marker="o", label=r"$K = 9$")
# ax1.plot(delta_B_array, alphas_mean_10[1:],marker="o", label=r"$K = 10$")

ax1.set_xlabel(r"$\Delta B$")
ax1.set_ylabel(r"$\alpha$")

#ax1.set_xscale('log')
#ax1.tick_params(axis='y', labelcolor="blue")
#ax1.legend(loc="upper right")

# Creating a second y-axis

# Plotting the second dataset on the right y-axis
#ax1.errorbar(delta_B_array, betas_mean, yerr=betas_sigma,marker="o", label=r"$\beta$", color="red")
#ax1.tick_params(axis='y', labelcolor="red")
#ax2.set_yticks(np.linspace(-1,1,5))
ax1.legend(loc="upper right",prop={'size': 20})
#ax1.set_yticks(np.linspace(0,1,5))


plt.show()