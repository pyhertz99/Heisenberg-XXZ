import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt
from matplotlib import rcParams
from heisenbergXXZ import *
from quantumChaos import *


# SYSTEM PARAMETERS

BOUNDARY = 1 #1 if the chain is connected at edges
N = 10 #number of cells
L = N//2 #number of excited cells (L <= N)
M = int(comb(N,L))

J_xy = 1.0 #coupling constant J_x = J_y

POINTS = 20 #number of J_z values
J_z_min = 0
J_z_max = 10
J_z_array = np.linspace(J_z_min,J_z_max,POINTS) #coupling constant J_z

B_site = 0 #site on which to insert B field
B_value = 1 #value of this B field


# COMPUTE SPECTRUM

#create magnetic field array
B = np.zeros(N)
B[B_site] = 1


spectra = np.zeros((POINTS,M))
for i in range(POINTS):
    
    J_z = J_z_array[i]
    print(J_z)
    
    #create connectivity matrices
    M_xy, M_z = createConnectivityMatrices(N, J_xy, J_z, boundary=BOUNDARY)
    
    #create transformation matrix
    PI, spin_indices = subspaceTransformationMatrix(N, L, M)
    
    #create hamiltonian
    H = createHamiltonian(N, PI, M_xy, M_z, B)
    
    #diagonalize
    spectrum, eigvecs = diagonalizeHamiltonian(H)
    
    spectra[i] = spectrum

#%% ANALYZE SPECTRUM

DEG = 7 #degree of polynom used for unfolding
BINS = 20 #number of bins in histogram

#number of energy levels to cut from each edge
#(could help when fit diverges)
CUT_EDGES = 10

betas = np.zeros((POINTS))
rs = np.zeros((POINTS))
for i in range(POINTS):
    spectrum = spectra[i]
    
    #compute Brody fit
    unfoldedSpectrum, polynom = unfoldSpectrum(spectrum, DEG, CUT_EDGES)
    level_dif = levelDifference(unfoldedSpectrum)
    hist, bin_centres, params, cov = brodyFit(level_dif, BINS)
    betas[i] = params[0]
    
    #compute mean ratio of consecutive levels
    r, sigma_r = ratioConsecutiveLevels(spectrum)
    rs[i] = r

betas_mean = np.zeros(POINTS)
rs_mean = np.zeros(POINTS)
for i in range(POINTS):
    betas_mean[i] = np.mean(betas[i])
    rs_mean[i] = np.mean(rs[i])

#%% PLOT HISTOGRAM

rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 30})

fig, ax1 = plt.subplots(1,1,figsize=(2*6,2*4))

# Plotting the first dataset on the left y-axis
ax1.plot(J_z_array, rs_mean, marker="o", label=r"$\langle r \rangle$", color="blue")
ax1.set_xlabel(r"$J_z$")
ax1.tick_params(axis='y', labelcolor="blue")
#ax1.set_yticks(np.linspace(0,0.4,5))´
#ax1.legend(loc="upper right")

# Creating a second y-axis
ax2 = ax1.twinx()

# Plotting the second dataset on the right y-axis
ax2.plot(J_z_array, betas_mean, marker="o", label=r"$\beta$", color="red")
ax2.tick_params(axis='y', labelcolor="red")
#ax2.set_yticks(np.linspace(-1,1,5))
#ax2.legend(loc="upper right")

fig.legend(bbox_to_anchor=(0.9, 0.88))

plt.show()