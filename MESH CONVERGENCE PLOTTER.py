#####mesh size convergence plotter
##### plotting code

import numpy as np
import matplotlib.pyplot as plt


###general parameters not mesh dependent-set as desired
epsilon = 1*(10**(-15)) #for use as small parameter
R = 2.5 # radius = half height of domain
width = 5 # width of domain
L = 0.25 # half thickness of membrane
n=20 # number of channels in the membrane

#distances from membrane to measure flux
dist_1= 0.5
dist_2 = 1.5  ##should be larger than dist_1
delta = width/n #channel spacing
eps = 0.1 #ratio of channel width to spacing, set value as desired
P = 1/((L/eps)+(2*delta/np.pi)*(np.log(1/(8*eps))+1)) # defines permeability in terms of channel geometry 
exact_flux = (P/(1 + 2*P*(R-L)))

meshlistmore = np.linspace(0.007,0.1,100)
exact_flux_list = [abs(exact_flux) for y in meshlistmore]



mesh_size_list = np.genfromtxt("mesh_size_array").tolist()
flux_full_above_1_list = np.genfromtxt("full_flux_above_1_array").tolist()
flux_num_above_1_list =np.genfromtxt("num_flux_above_1_array").tolist()
flux_an_above_1_list = np.genfromtxt("an_flux_above_1_array").tolist()




flux_error_1_above_f =[abs(flux_full_above_1_list[x]-flux_full_above_1_list[-1])/flux_full_above_1_list[-1]for x in range(len(flux_full_above_1_list))]
flux_error_1_above_num =[abs(flux_num_above_1_list[x]-flux_num_above_1_list[-1])/flux_num_above_1_list[-1] for x in range(len(flux_num_above_1_list))]
flux_error_1_above_an =[abs(flux_an_above_1_list[x]-flux_an_above_1_list[-1])/flux_an_above_1_list[-1] for x in range(len(flux_an_above_1_list))]



import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


# LaTeX + font settings
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 12,
})

# Create figure and axis
fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))

# ax.plot(mesh_size_list, flux_error_1_above_an,
#         label='analytical effective interface',
#         linestyle='--', marker='o', color='red', markersize=4)

ax.plot(mesh_size_list, flux_error_1_above_f,
        label='full simulation',
        linestyle='dotted', marker='x', color='indigo', markersize=4)

ax.plot(mesh_size_list, flux_error_1_above_num,
        label='numerical effective interface',
        linestyle='--', marker='s', color='turquoise', markersize=4)

# Axis labels
ax.set_xlabel('Mesh size')
ax.set_ylabel('Relative flux error')

# Y-axis formatting
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
ax.yaxis.get_major_formatter().set_scientific(True)
ax.yaxis.get_major_formatter().set_useOffset(False)
ax.ticklabel_format(axis='y', style='sci', scilimits=(-5, -5))

# Legend and grid
ax.legend(fontsize=10)
ax.grid(True, linestyle='--', alpha=0.6)

# Optional title
# ax.set_title(r'Flux error vs mesh size', fontsize=16)

# Final layout
fig.tight_layout()
fig.savefig('flux_error_vs_meshsize.eps', bbox_inches='tight', dpi=500)
# plt.show()
# plt.close()
