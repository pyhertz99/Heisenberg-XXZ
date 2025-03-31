"""
Analyze spectra from saved file for 
discrete finite range topology with respect
to K.
"""

import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt
from matplotlib import rcParams
from heisenbergXXZ import *
import connectivityMatrices as cm
from quantumChaos import *

spectra = np.load("./data/discFiniteRange/discFiniteRange-XYZ1-smp20/spectra-N12.npy")

N = 12 #number of cells
L = N//2 #number of excited cells (L <= N)
M = int(comb(N,L))

J_xy = 1.0 #coupling constant J_x = J_y
J_z = 1.0

B_0 = 0.0 #magnetic field mean value
delta_Bs = np.array([0.1,0.2,0.5,1.0,1.5,2.0])
nB = delta_Bs.size

samples = 20

POINTS = N//2
SAMPLES = samples

DEG = 7 #degree of polynom used for unfolding
BINS = 100 #number of bins in histogram

#number of energy levels to cut from each edge
#(could help when fit diverges)
CUT_EDGES = 10

betas = np.zeros((nB,POINTS,SAMPLES))
rs = np.zeros((nB,POINTS,SAMPLES))
for n in range(nB):
    print(n)
    for i in range(POINTS):
        for j in range(SAMPLES):
            spectrum = spectra[n,i,j]
            
            #compute Brody fit
            unfoldedSpectrum, polynom = unfoldSpectrum(spectrum, DEG, CUT_EDGES)
            level_dif = levelDifference(unfoldedSpectrum)
            hist, bin_centres, params, cov = brodyFit(level_dif, BINS)
            betas[n,i,j] = params[0]
            
            #compute mean ratio of consecutive levels
            r, sigma_r = ratioConsecutiveLevels(spectrum)
            rs[n,i,j] = r

#%%
r_0 = 2*np.log(2) - 1
r_1 = 4 - 2*np.sqrt(3)

alphas = (rs - r_0)/(r_1-r_0)

betas_mean = np.zeros((nB,POINTS))
betas_sigma = np.zeros((nB,POINTS))
alphas_mean = np.zeros((nB,POINTS))
alphas_sigma = np.zeros((nB,POINTS))
for n in range(nB):
    for i in range(POINTS):
        betas_mean[n,i] = np.mean(betas[n,i])
        betas_sigma[n,i] = np.std(betas[n,i])
        alphas_mean[n,i] = np.mean(alphas[n,i])
        alphas_sigma[n,i] = np.std(alphas[n,i])
    
#%%

rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 40})

fig, axs = plt.subplots(1,2,figsize=(3*6,2*4),dpi=300)

for i in range(1,nB):
    dB = delta_Bs[i]
    axs[0].errorbar(np.arange(1,N//2 + 1), alphas_mean[i],yerr=alphas_sigma[i],marker="o",markersize=10,label=r"$\Delta B = $"+str(dB))
    axs[1].errorbar(np.arange(1,N//2 + 1), betas_mean[i],yerr=betas_sigma[i],marker="o",markersize=10,label=r"$\Delta B = $"+str(dB))
#ax1.plot(delta_B_array, betas_mean,marker="o", label=r"$K = 1$")

# ax1.plot(delta_B_array, alphas_mean_3[1:],marker="o", label=r"$K = 3$")
# ax1.plot(delta_B_array, alphas_mean_5[1:],marker="o", label=r"$K = 5$")
# ax1.plot(delta_B_array, alphas_mean_7[1:],marker="o", label=r"$K = 7$")
# ax1.plot(delta_B_array, alphas_mean_9[1:],marker="o", label=r"$K = 9$")
# ax1.plot(delta_B_array, alphas_mean_10[1:],marker="o", label=r"$K = 10$")

axs[0].set_xlabel(r"$K$")
axs[1].set_xlabel(r"$K$")
axs[0].set_ylabel(r"$\alpha$")
axs[1].set_ylabel(r"$\beta$")

#ax1.set_xscale('log')
#ax1.tick_params(axis='y', labelcolor="blue")
#ax1.legend(loc="upper right")

# Creating a second y-axis

# Plotting the second dataset on the right y-axis
#ax1.errorbar(delta_B_array, betas_mean, yerr=betas_sigma,marker="o", label=r"$\beta$", color="red")
#ax1.tick_params(axis='y', labelcolor="red")
#axs[0].set_yticks([0.0,0.2,0.4,0.6,0.8,1.0])
axs[1].set_yticks([-0.5,0,0.5,1])
axs[0].set_xticks(range(1,8))
axs[1].set_xticks(range(1,8))
#axs[1].set_yticks([-1,-0.5,0,0.5,1.0])
axs[0].legend(prop={'size': 20})
#ax1.set_yticks(np.linspace(0,1,5))

plt.tight_layout()


plt.show()