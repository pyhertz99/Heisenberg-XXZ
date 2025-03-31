"""
Plot saved entropy contour plots.
"""

import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt
from matplotlib import rcParams
from heisenbergXXZ import *
import lowMagnetization as lm
import connectivityMatrices as cm

entropy_arrays1 = np.load("./data/discFiniteRange/entropyContourPlot/K1-t20-N100.npy")
entropy_arrays2 = np.load("./data/discFiniteRange/entropyContourPlot/K1-t20-dB1-2-N100.npy")
entropy_arrays3 = np.load("./data/discFiniteRange/entropyContourPlot/K2-t20-N100.npy")


# PARAMETERS

N = 100 #number of cells
L = 1 #number of excited cells (L <= N)
K = 1 #degree of connectivity
M = int(comb(N,L))

J_xy = 1.0 #coupling constant J_x = J_y
J_z = 1.0 #coupling constant J_z

B_0 = 0.0 #magnetic field mean value
delta_B = 1.0 #magnetic field spread

t_max = 20
t_steps = 100

#%% PLOTTING
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 55})

fig, axs = plt.subplots(2,3,figsize=(4.5*6,4.2*4),layout="constrained",dpi=200)
axs[0,0].imshow(np.flip(entropy_arrays1,axis=0), aspect="auto", interpolation="none", extent=[80,120,0,t_max],vmax=0.6)
axs[0,1].imshow(np.flip(entropy_arrays2,axis=0), aspect="auto", interpolation="none", extent=[80,120,0,t_max],vmax=0.6)
plot1 = axs[0,2].imshow(np.flip(entropy_arrays3,axis=0), aspect="auto", interpolation="none", extent=[80,120,0,t_max],vmax=0.6)

axs[1,0].imshow(np.flip(img1,axis=1), aspect="auto", interpolation="none", extent=[80,120,0,t_max],vmin=0.6)
axs[1,1].imshow(np.flip(img2,axis=1), aspect="auto", interpolation="none", extent=[80,120,0,t_max],vmin=0.6)
plot2 = axs[1,2].imshow(np.flip(img3,axis=1), aspect="auto", interpolation="none", extent=[80,120,0,t_max],vmin=0.6)

plt.rcParams['axes.titley'] = -0.1    # y is in axes-relative coordinates.
plt.rcParams['axes.titlepad'] = 0
axs[1,0].set_title("a)", fontsize=55)
axs[1,1].set_title("b)",  fontsize=55)
axs[1,2].set_title("c)",  fontsize=55)
axs[0,1].set_yticklabels([])
axs[0,2].set_yticklabels([])
axs[1,1].set_yticklabels([])
axs[1,2].set_yticklabels([])
axs[0,0].set_xticks([])
axs[0,1].set_xticks([])
axs[0,2].set_xticks([])
axs[1,0].set_xticks([])
axs[1,1].set_xticks([])
axs[1,2].set_xticks([])

axs[0,0].set_ylabel(r'$t$')
axs[1,0].set_ylabel(r'$t$')
fig.colorbar(plot1,ax=axs[0,:])
fig.colorbar(plot2,ax=axs[1,:])
#fig.tight_layout()
plt.show()

