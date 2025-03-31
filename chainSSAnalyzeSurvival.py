"""
Reads saved samples of sinlge-spin state evolution for given
parameters and plots mean survival probability
as a function of J_z.
"""

import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt
from matplotlib import rcParams
from heisenbergXXZ import *

# PARAMETERS

N = 11 #number of cells
L = 1 #number of excited cells (L <= N)
M = int(comb(N,L))

J_xy = 1.0 #coupling constant J_x = J_y
J_z_array = np.arange(0.0,2.1,0.1) #coupling constant J_z

B_0 = 0.0 #magnetic field mean value
delta_B_array = [0,0.2,0.5,1,2,5] #magnetic field spread

t_max = 200
t_steps = 100

folder = "./data/"

len_delta_B_array = len(delta_B_array)
len_J_z_array = len(J_z_array)

#specify how many samples were created
samples_array = np.full((len_delta_B_array,len_J_z_array),50) #50 for dB != 0
samples_array[0] = np.full((len_J_z_array),1) #just one for dB=0


# COMPUTATION

survival_means_plot_ss = np.full((len(delta_B_array),len(J_z_array)),0.0)

for i in range(len(J_z_array)):
    J_z = J_z_array[i]
    J_z_string = f"{J_z:.2f}"
    print("J_z = " + J_z_string)
    
    for j in range(len(delta_B_array)):
        delta_B = delta_B_array[j]
        delta_B_string = f"{delta_B:.2f}"
        print("dB = " + delta_B_string)
        
        samples = samples_array[j,i]
        means = np.full(samples,0.0)
        for sample in range(samples):
            psi_array = np.load(folder + "single-spin/N" + str(N) + "-Z" 
                                + J_z_string + "-dB" + delta_B_string
                                + "-tMax" + str(t_max) + "-tSteps" 
                                + str(t_steps) + "-" + str(sample)
                                + ".npy")
            survival_array = survivalProbability(psi_array, t_steps)
            means[sample] = np.mean(survival_array)
            
        survival_means_plot_ss[j,i] = np.mean(means)

#%% PLOTTING

rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 30})


fig, ax = plt.subplots(1,1, figsize=(2*6,2*4), dpi=300)

for i in range(len(delta_B_array)):
    delta_B = delta_B_array[i]
    delta_B_string = f"{delta_B:.2f}"
    plt.plot(J_z_array, survival_means_plot_ss[i], linestyle="-", marker="o", label=r'$\Delta B =$' + delta_B_string)

ax.set_xlabel(r'$J_z$')
ax.set_ylabel(r'$\left< P \right>$')
ax.set_ylim(0.0,1.0)
ax.legend(prop={'size': 15},loc="upper left")

plt.show()
                
