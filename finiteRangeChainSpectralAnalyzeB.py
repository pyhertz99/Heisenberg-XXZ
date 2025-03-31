"""
Compute spectral characteristics of
system with continuous finite range
topology with respect to B.
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
J_z = 1.0 #coupling constant J_z

SAMPLES = 5 #number of samples for each parameter setting
POINTS = 11 #number of q values

q_min = 0.0
q_max = 1.0

q_array = np.linspace(q_min,q_max,POINTS) #coupling constant J_z


B_0 = 0.0 #magnetic field mean value
delta_B = 100.0 #magnetic field spread


# COMPUTE SPECTRUM

spectra = np.zeros((POINTS,SAMPLES,M))
for i in range(POINTS):
    
    q = q_array[i]
    print(q)
    for j in range(SAMPLES):
        
        #create random magnetic field array
        B = np.random.uniform(low=B_0-delta_B,high=B_0+delta_B,size=N)
        
        #create connectivity matrices
        M_xy, M_z = cm.chainFiniteRange(N, J_xy, J_z, q)
        
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

#%% PLOT

rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 30})

fig, ax1 = plt.subplots(1,1,figsize=(2*6,2*4))

# Plotting the first dataset on the left y-axis
ax1.errorbar(np.linspace(q_min,q_max,POINTS), alphas_mean, yerr=alphas_sigma,marker="o", label=r"$\alpha$", color="blue")
ax1.set_xlabel(r"$q$")
#ax1.tick_params(axis='y', labelcolor="blue")
#ax1.legend(loc="upper right")

# Creating a second y-axis

# Plotting the second dataset on the right y-axis
ax1.errorbar(np.linspace(q_min,q_max,POINTS), betas_mean, yerr=betas_sigma,marker="o", label=r"$\beta$", color="red")
#ax1.tick_params(axis='y', labelcolor="red")
#ax2.set_yticks(np.linspace(-1,1,5))
#ax2.legend(loc="upper right")
ax1.set_yticks(np.linspace(0,1,5))

fig.legend(bbox_to_anchor=(0.9, 0.88))

plt.show()

#%% PLOT ALL

rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 35})

fig, ax = plt.subplots(1,1,figsize=(2*6,2*4))

xs = np.linspace(0.0,1.0,POINTS)
#ax.plot(np.arange(1,POINTS+1), alphas_mean_01,marker="o")
#ax.plot(xs, betas_mean_02,marker="o",label=r"$\Delta B = 0.2$", markersize=9)
ax.plot(xs, betas_mean_05,marker="o",label=r"$\Delta B = 0.5$",markersize=9)
ax.plot(xs, betas_mean_10,marker="o",label=r"$\Delta B = 1.0$",markersize=9)
ax.plot(xs, betas_mean_20,marker="o",label=r"$\Delta B = 2.0$",markersize=9)
ax.plot(xs, betas_mean_50,marker="o",label=r"$\Delta B = 5.0$",markersize=9)
ax.plot(xs, betas_mean_100,marker="o",label=r"$\Delta B = 10.0$",markersize=9)
ax.plot(xs, betas_mean_200,marker="o",label=r"$\Delta B = 20.0$",markersize=9)
ax.plot(xs, betas_mean_1000,marker="o",label=r"$\Delta B = 100.0$",markersize=9)
ax.set_xlabel(r"$q$")
ax.set_ylabel(r"$\beta$")
#ax1.tick_params(axis='y', labelcolor="blue")
#ax1.legend(loc="upper right")

#ax1.errorbar(np.arange(1,POINTS+1), betas_mean, yerr=betas_sigma,marker="o", label=r"$\beta$", color="red")
#ax1.tick_params(axis='y', labelcolor="red")
#ax2.set_yticks(np.linspace(-1,1,5))
#ax2.legend(loc="upper right")
#ax.set_xticks(np.linspace(1,7,7))

fig.legend(bbox_to_anchor=(0.90, 0.88),prop={'size': 13})

plt.show()
