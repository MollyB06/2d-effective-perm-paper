####time step convergence plotting 
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np



full_flux_above_list_2_5 = np.genfromtxt('TCfull_flux_above_array').tolist()
full_flux_below_list_2_5 =np.genfromtxt('TCfull_flux_below_array').tolist()
hom_flux_above_list_2_5 = np.genfromtxt('TChom_flux_above_array').tolist()
hom_flux_below_list_2_5 = np.genfromtxt('TChom_flux_below_array').tolist()
# delt_list = np.genfromtxt("delt_array").tolist()
delt_list = [0.05,0.1,0.25,0.5,1.25,2.5]

###plotting part
rel_full_flux_above_2_5=[abs(full_flux_above_list_2_5[i]-full_flux_above_list_2_5[0])/full_flux_above_list_2_5[0] for i in range(len(delt_list))]
rel_full_flux_below_2_5=[abs(full_flux_below_list_2_5[i]-full_flux_below_list_2_5[0])/full_flux_below_list_2_5[0] for i in range(len(delt_list))]
rel_hom_flux_above_2_5=[abs(hom_flux_above_list_2_5[i]-hom_flux_above_list_2_5[0])/hom_flux_above_list_2_5[0] for i in range(len(delt_list))]
rel_hom_flux_below_2_5=[abs(hom_flux_below_list_2_5[i]-hom_flux_below_list_2_5[0])/hom_flux_below_list_2_5[0] for i in range(len(delt_list))]









# Set LaTeX and font settings
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
fig, ax = plt.subplots(1, 1, figsize=(6, 4))

# Plot curves
ax.plot(delt_list, rel_full_flux_above_2_5, label='full simulation', linestyle='-', marker='s', color='orange')
ax.plot(delt_list, rel_hom_flux_above_2_5, label='effective interface', linestyle='-', marker='s', color='green')

# Axis labels
ax.set_xlabel('Time step size')
ax.set_ylabel('Relative flux error')

# Y-axis formatting
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
ax.yaxis.get_major_formatter().set_scientific(False)
ax.yaxis.get_major_formatter().set_useOffset(False)

# Legend and grid
ax.legend(fontsize=10)
ax.grid(True, linestyle='--', alpha=0.6)

# Optional: add title
# ax.set_title(r'Relative error in flux at time $t = 2.5$ vs time step size', fontsize=16)

# Final layout and save or display
fig.tight_layout()
fig.savefig('flux_error_vs_timestep.eps', bbox_inches='tight', dpi=500)
# plt.show()
# plt.close()


    
