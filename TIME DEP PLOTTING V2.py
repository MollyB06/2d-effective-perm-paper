###fixed plots
##time dep plots
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np

#domain parameters for permabiltiy calcs
R = 2.5 # radius = half height of domain
width = 5
L = 0.25 #half length of channel

n_chunk = 20# number of "impermeable chunks" of membrane not including endpoints, # of channels = n-1 #### change this to have varying permeability
delta = width/n_chunk 


###larger permeability
eps = 0.25
P = 1/((L/eps)+(2*delta/np.pi)*(np.log(1/(8*eps))+1)) # defines permeability in terms of channel geometry 


##smaller permeability
eps2 = 0.05
P2 = 1/((L/eps2)+(2*delta/np.pi)*(np.log(1/(8*eps2))+1))

##read in data




times_list_roundlong= np.genfromtxt('times_list_long_1').tolist()
full_flux_above_listlong=np.genfromtxt('full_flux_above_array_long_1').tolist()
full_flux_below_listlong=np.genfromtxt('full_flux_below_array_long_1').tolist()
hom_flux_above_listlong=np.genfromtxt('newhom_flux_above_array_long_1').tolist()
hom_flux_below_listlong=np.genfromtxt('newhom_flux_below_array_long_1').tolist()





error_above_num_vs_fulllong=[abs(hom_flux_above_listlong[i] -full_flux_above_listlong[i]) for i in range(len(hom_flux_above_listlong))]
error_below_num_vs_fulllong=[hom_flux_below_listlong[i] -full_flux_below_listlong[i] for i in range(len(hom_flux_above_listlong))]


times_list_roundshort = np.genfromtxt("times_list_short_1").tolist()
full_flux_above_list = [full_flux_above_listlong[i] for i in range(len(times_list_roundshort))]
full_flux_below_list = [full_flux_below_listlong[i] for i in range(len(times_list_roundshort))]

EM_flux_above_list = np.genfromtxt('newEM_flux_above_array_1').tolist()
EM_flux_below_list =np.genfromtxt('newEM_flux_below_array_1').tolist()
error_above_full_vs_EM = [abs(EM_flux_above_list[i]-full_flux_above_list[i]) for i in range(len(times_list_roundshort))]
error_below_full_vs_EM =[abs(EM_flux_below_list[i]-full_flux_below_list[i]) for i in range(len(times_list_roundshort))]


###smaller permeability run

hom_flux_above_listlong2=np.genfromtxt('newhom_flux_above_array_long_2').tolist()
hom_flux_below_listlong2=np.genfromtxt('newhom_flux_below_array_long_2').tolist()
full_flux_above_listlong2=np.genfromtxt('full_flux_above_array_long_2').tolist()
full_flux_below_listlong2=np.genfromtxt('full_flux_below_array_long_2').tolist()


error_above_num_vs_full2=[abs(hom_flux_above_listlong2[i] -full_flux_above_listlong2[i]) for i in range(len(times_list_roundlong))]
error_below_num_vs_full2=[abs(hom_flux_below_listlong2[i] -full_flux_below_listlong2[i]) for i in range(len(times_list_roundlong))]

full_flux_above_list2 = [full_flux_above_listlong2[i] for i in range(len(times_list_roundshort))]
full_flux_below_list2 =  [full_flux_below_listlong2[i] for i in range(len(times_list_roundshort))]



EM_flux_above_list2 = np.genfromtxt('newEM_flux_above_array_2').tolist()
EM_flux_below_list2 =np.genfromtxt('newEM_flux_below_array_2').tolist()

error_above_full_vs_EM2 = [abs(EM_flux_above_list2[i]-full_flux_above_list2[i]) for i in range(len(times_list_roundshort))]
error_below_full_vs_EM2 =[abs(EM_flux_below_list2[i]-full_flux_below_list2[i]) for i in range(len(times_list_roundshort))]














error_above_num_vs_EM = [abs(hom_flux_above_listlong[i]-EM_flux_above_list[i]) for i in range(len(EM_flux_above_list))]
error_below_num_vs_EM = [abs(hom_flux_below_listlong[i]-EM_flux_below_list[i]) for i in range(len(EM_flux_below_list))]

relative_error_aboveNF =[error_above_num_vs_fulllong[i]/full_flux_above_listlong[i] for i in range(len(full_flux_above_listlong))]
relative_error_belowNF =[error_below_num_vs_fulllong[i]/full_flux_below_listlong[i] for i in range(len(full_flux_below_listlong))]

relative_error_aboveNE =[error_above_num_vs_EM[i]/full_flux_above_list[i] for i in range(len(EM_flux_above_list))]
relative_error_belowNE =[error_below_num_vs_EM[i]/full_flux_below_list[i] for i in range(len(full_flux_below_list))]

relative_error_aboveEF =[error_above_full_vs_EM[i]/full_flux_above_list[i] for i in range(len(EM_flux_above_list))]
relative_error_belowEF =[error_below_full_vs_EM[i]/full_flux_below_list[i] for i in range(len(full_flux_below_list))]



error_above_num_vs_EM2 = [abs(hom_flux_above_listlong2[i]-EM_flux_above_list2[i]) for i in range(len(EM_flux_above_list2))]
error_below_num_vs_EM2 = [abs(hom_flux_below_listlong2[i]-EM_flux_below_list2[i]) for i in range(len(EM_flux_below_list2))]

relative_error_aboveNF2 =[error_above_num_vs_full2[i]/full_flux_above_listlong2[i] for i in range(len(full_flux_above_listlong2))]
relative_error_belowNF2 =[error_below_num_vs_full2[i]/full_flux_below_listlong2[i] for i in range(len(full_flux_below_listlong2))]

relative_error_aboveNE2 =[error_above_num_vs_EM2[i]/full_flux_above_list2[i] for i in range(len(EM_flux_above_list2))]
relative_error_belowNE2 =[error_below_num_vs_EM2[i]/full_flux_below_list2[i] for i in range(len(full_flux_below_list2))]

relative_error_aboveEF2 =[error_above_full_vs_EM2[i]/full_flux_above_list2[i] for i in range(len(EM_flux_above_list2))]
relative_error_belowEF2 =[error_below_full_vs_EM2[i]/full_flux_below_list2[i] for i in range(len(full_flux_below_list2))]


steady_flux_list =[(P/(1 + 2*P*(R-L))) for i in range(len(full_flux_above_list))]
steady_flux_listlong =[(P/(1 + 2*P*(R-L))) for i in range(len(full_flux_above_listlong))]

steady_flux_listlong2 =[(P2/(1 + 2*P2*(R-L))) for i in range(len(full_flux_above_listlong))]



import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter


plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
plt.rcParams.update({
    "text.usetex": True,  # Enable full LaTeX rendering
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],  # Optional, reinforces LaTeX look
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 30,
})


import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

# Create figure and axis with specified size
fig, ax = plt.subplots(1, 1, figsize=(6, 4))

# Plot the first group (p = 0.388)
line1 = ax.plot(times_list_roundlong, full_flux_above_listlong, label='full simulation', linestyle='dashed', marker='x', color='red',markersize=3.8)[0]
line2 = ax.plot(times_list_roundlong, hom_flux_above_listlong, label='effective interface problem', linestyle='dashed', marker='s', markerfacecolor='none', color='red',markersize=3.8)[0]
line3 = ax.plot(times_list_roundshort, EM_flux_above_list, label='Euler-Maclaurin approximation', linestyle='dashed', marker='o', color='red',markersize=3.8)[0]
line4 = ax.plot(times_list_roundlong, steady_flux_listlong, label='exact effective steady flux', linestyle='-', color='red',markersize=3.8)[0]

# Plot the second group (p = 0.955)
line5 = ax.plot(times_list_roundlong, full_flux_above_listlong2, label='full simulation', linestyle='dashed', marker='x', color='navy',markersize=3.8)[0]
line6 = ax.plot(times_list_roundlong, hom_flux_above_listlong2, label='effective interface problem', linestyle='dashed', marker='s', markerfacecolor='none',color='navy',markersize=3.8)[0]
line7 = ax.plot(times_list_roundshort, EM_flux_above_list2, label='Euler-Maclaurin approximation', linestyle='dashed', marker='o', color='navy',markersize=3.8)[0]
line8 = ax.plot(times_list_roundlong, steady_flux_listlong2, label='exact effective steady flux', linestyle='-', color='navy',markersize=3.8)[0]

# Create dummy handles for group headers
group1_label = Line2D([], [], color='none', label=r'$P_\text{eff}$ = 0.953')
group2_label = Line2D([], [], color='none', label=r'$P_\text{eff}$ = 0.189')


#Build the custom legend
handles = [
    group1_label, line1, line2, line3, line4,
    group2_label, line5, line6, line7, line8
]
labels = [
    r'$P_\text{eff}$ = 0.953',
    '    full simulation',
    '    effective interface problem',
    '    Euler-Maclaurin approximation',
    '    exact effective steady flux',
    r'$P_\text{eff}$ = 0.189',
    '    full simulation',
    '    effective interface problem',
    '    Euler-Maclaurin approximation',
    '    exact effective steady flux'
]


# Add legend
ax.legend(handles, labels, fontsize=6)

# Axis labels
ax.set_xlabel('Time')
ax.set_ylabel('Flux')

# Format y-axis
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
ax.yaxis.get_major_formatter().set_scientific(False)
ax.yaxis.get_major_formatter().set_useOffset(False)
ax.set_ylim(0, 0.2)

# Grid
ax.grid(True, linestyle='--', alpha=0.6)

# Save figure
fig.tight_layout()
fig.savefig('fixed_flux_vs_timedashed.eps', bbox_inches='tight', dpi=500)
plt.close()


import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter



# Create figure and axis
fig, ax = plt.subplots(1, 1, figsize=(6, 4))


# Plot group 1 (p = 0.388)
line1 = ax.plot(times_list_roundlong, error_above_num_vs_fulllong, label='effective vs full', linestyle='dashed', marker='x', color='red',markersize=3.8)[0]
line2 = ax.plot(times_list_roundshort, error_above_num_vs_EM, label='effective vs Euler-Maclaurin', linestyle='dashed', marker='o', color='red',markersize=3.8)[0]
line3 = ax.plot(times_list_roundshort, error_above_full_vs_EM, label='Euler-Maclaurin vs full', linestyle='dashed', marker='s',markerfacecolor='none', color='red',markersize=3.8)[0]

# Plot group 2 (p = 0.955)
line4 = ax.plot(times_list_roundlong, error_above_num_vs_full2, label='effective vs full', linestyle='dashed', marker='x', color='navy',markersize=3.8)[0]
line5 = ax.plot(times_list_roundshort, error_above_num_vs_EM2, label='effective vs Euler-Maclaurin', linestyle='dashed', marker='o', color='navy',markersize=3.8)[0]
line6 = ax.plot(times_list_roundshort, error_above_full_vs_EM2, label='Euler-Maclaurin vs full', linestyle='dashed', marker='s',markerfacecolor='none', color='navy',markersize=3.8)[0]


inset_ax= ax.inset_axes([.6, .3, .35, .35]) # [x, y, width, height] w.r.t. ax # setsup inset axes

# # Plot group 1 (p = 0.388)
inset_ax.plot(times_list_roundlong, relative_error_aboveNF, label='effective vs full', linestyle='dashed', marker='x', color='red',markersize=3)[0]
inset_ax.plot(times_list_roundshort, relative_error_aboveNE, label='effective vs Euler-Maclaurin', linestyle='dashed', marker='o', color='red',markersize=3)[0]
inset_ax.plot(times_list_roundshort, relative_error_aboveEF, label='Euler-Maclaurin vs full', linestyle='dashed', marker='s',markerfacecolor='none', color='red',markersize=3)[0]

# Plot group 2 (p = 0.955)
inset_ax.plot(times_list_roundlong, relative_error_aboveNF2, label='effective vs full', linestyle='dashed', marker='x', color='navy',markersize=3)[0]
inset_ax.plot(times_list_roundshort, relative_error_aboveNE2, label='effective vs Euler-Maclaurin', linestyle='dashed', marker='o', color='navy',markersize=3)[0]
inset_ax.plot(times_list_roundshort, relative_error_aboveEF2, label='Euler-Maclaurin vs full', linestyle='dashed', marker='s',markerfacecolor='none', color='navy',markersize=3)[0]
inset_ax.set_xlim(0, 5)
inset_ax.set_ylabel(r'Relative flux error',fontsize=10)
inset_ax.tick_params(axis='both', labelsize=10) 



# Create dummy handles for group headers
group1_label = Line2D([], [], color='none', label=r'$P_\text{eff}$ = 0.953')
group2_label = Line2D([], [], color='none', label=r'$P_\text{eff}$ = 0.189')

# Custom legend with group headers
handles = [
    group1_label, line1, line2, line3,
    group2_label, line4, line5, line6
]
#Custom legend with group headers
labels = [
    r'$P_\text{eff}$ = 0.953',
    '    effective vs full',
    '    effective vs Euler-Maclaurin',
    '    Euler-Maclaurin vs full',
    r'$P_\text{eff}$ = 0.189',
    '    effective vs full',
    '    effective vs Euler-Maclaurin',
    '    Euler-Maclaurin vs full'
]


# Axis labels and formatting
ax.set_xlabel('Time')
# ax.set_ylabel(r'Relative flux error')
ax.set_ylabel(r'Absolute flux error')


ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
ax.yaxis.get_major_formatter().set_scientific(False)
ax.yaxis.get_major_formatter().set_useOffset(False)
#ax.set_ylim(0, 0.6)

# Add legend and grid
ax.legend(handles, labels, fontsize=6)
ax.grid(True, linestyle='--', alpha=0.6)

# Optional: title
# ax.set_title(r'Relative flux error vs time for $p=$' + str(round(P, 3)), fontsize=16)

# Finalize and save (or show)
fig.tight_layout()
fig.savefig('fixed_abs_flux_errornoinset.eps', bbox_inches='tight', dpi=500)
plt.close()




