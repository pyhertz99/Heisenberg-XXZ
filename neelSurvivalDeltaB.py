"""
Reads saved samples of Neel state evolution for given
parameters and plots mean survival probability
as a function of t for all dB.
"""

import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt
from matplotlib import rcParams
from heisenbergXXZ import *

# PARAMETERS

N = 10 #number of cells
L = N//2 #number of excited cells (L <= N)
M = int(comb(N,L))

J_xy = 1.0 #coupling constant J_x = J_y
J_z = 2.0 #coupling constant J_z

B_0 = 0.0 #magnetic field mean value
delta_B_array = [0,0.2,0.5,1,2,5] #magnetic field spread

t_max = 10
t_steps = 100

folder = "./data/"

len_delta_B_array = len(delta_B_array)

#specify how many samples were created
samples_array = np.full((len_delta_B_array),50) #50 for dB != 0
samples_array[0] = 1 #just one for dB=0


# COMPUTATION

survival_arrays_plot = np.full((len_delta_B_array,t_steps),0.0)
J_z_string = f"{J_z:.2f}"

for j in range(len_delta_B_array):
    delta_B = delta_B_array[j]
    delta_B_string = f"{delta_B:.2f}"
    print("dB = " + delta_B_string)
    
    samples = samples_array[j]
    survival_arrays = np.full((samples,t_steps),0.0)
    for sample in range(samples):
        psi_array = np.load(folder + "neel/N" + str(N) + "-Z" 
                            + J_z_string + "-dB" + delta_B_string
                            + "-tMax" + str(t_max) + "-tSteps" 
                            + str(t_steps) + "-" + str(sample)
                            + ".npy")
        survival_arrays[sample] = survivalProbability(psi_array, t_steps)
    
    for i in range(t_steps):
        t_slice = survival_arrays[:,i]
        survival_arrays_plot[j,i] = np.mean(t_slice)

#%% PLOTTING

rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 30})

ts = np.linspace(0,t_max,t_steps)

fig, ax = plt.subplots(1,1, figsize=(2*6,2*4), dpi=300)

for i in range(len(delta_B_array)):
    delta_B = delta_B_array[i]
    delta_B_string = f"{delta_B:.2f}"
    plt.plot(ts, survival_arrays_plot[i], linestyle="-", label="dB = " + delta_B_string)

ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$\left< \psi(t) | \psi_0 \right>$')
ax.set_ylim(0,1)
ax.legend(prop={'size': 15})

plt.show()
