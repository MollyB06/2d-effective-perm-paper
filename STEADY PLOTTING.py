from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np

n=20 # number of "impermeable chunks" of membrane not including endpoints, # of channels = n-1
R = 2.5 # radius = half height of domain
width = 5 #width of domain
delta = width/n #channel spacing
L = 0.25 #half channel length
epsilon = 1*(10**(-15)) # to use as small number

#distances from membrane to compute flux
dist_1= 0.5
dist_2 = 1.5

flux_num_above_1_list = np.genfromtxt("flux_num_above_1_list_array").tolist()
flux_num_below_1_list=np.genfromtxt("flux_num_below_1_list_array").tolist()
flux_full_above_1_list = np.genfromtxt("flux_full_squish_above_1_list_array").tolist()
flux_full_below_1_list=np.genfromtxt("flux_full_squish_below_1_list_array").tolist()
flux_an_above_1_list = np.genfromtxt("flux_an_above_1_list_array").tolist()
flux_an_below_1_list =np.genfromtxt("flux_an_below_1_list_array").tolist()
er_num_v_an_1_above = np.genfromtxt("er_num_v_an_1_above_array").tolist()
er_num_v_an_1_below = np.genfromtxt("er_num_v_an_1_below_array").tolist()
er_num_v_full_1_above = np.genfromtxt("er_num_v_full_1_above_array").tolist()
er_num_v_full_1_below =np.genfromtxt("er_num_v_full_1_below_array").tolist()
er_an_v_full_1_above = np.genfromtxt("er_an_v_full_1_above_array").tolist()
er_an_v_full_1_below = np.genfromtxt("er_an_v_full_1_below_array").tolist()
#
flux_num_above_2_list = np.genfromtxt("flux_num_above_2_list_array").tolist()
flux_num_below_2_list=np.genfromtxt("flux_num_below_2_list_array").tolist()
flux_full_above_2_list = np.genfromtxt("flux_full_squish_above_2_list_array").tolist()
flux_full_below_2_list=np.genfromtxt("flux_full_squish_below_2_list_array").tolist()
flux_an_above_2_list = np.genfromtxt("flux_an_above_2_list_array").tolist()
flux_an_below_2_list =np.genfromtxt("flux_an_below_2_list_array").tolist()
er_num_v_an_2_above = np.genfromtxt("er_num_v_an_2_above_array").tolist()
er_num_v_an_2_below = np.genfromtxt("er_num_v_an_2_below_array").tolist()
er_num_v_full_2_above = np.genfromtxt("er_num_v_full_2_above_array").tolist()
er_num_v_full_2_below =np.genfromtxt("er_num_v_full_2_below_array").tolist()
er_an_v_full_2_above = np.genfromtxt("er_an_v_full_2_above_array").tolist()
er_an_v_full_2_below = np.genfromtxt("er_an_v_full_2_below_array").tolist()
permeability_list = np.genfromtxt("permeability_array").tolist()

################### PLOTTING CODE ####################################################
exactplist =np.linspace(0.1,1.7,100)
exact_flux_list = [(P/(1 + 2*P*(R-L))) for P in exactplist]
exact_flux2 =[(P/(1 + 2*P*(R-L))) for P in permeability_list]


er_num_v_exact_above =[abs(flux_num_above_1_list[i]-exact_flux2[i]) for i in range(len(exact_flux2))]
er_full_v_exact_above =[abs(flux_full_above_1_list[i]-exact_flux2[i]) for i in range(len(exact_flux2))]

rel_er_num_v_exact_above=[abs(er_num_v_exact_above[i]/flux_an_above_1_list[i]) for i in range(len(flux_an_above_1_list))]
rel_er_full_v_exact_above=[abs(er_full_v_exact_above[i]/flux_an_above_1_list[i]) for i in range(len(flux_an_above_1_list))]

rel_er_num_v_an_1_above = [abs(er_num_v_an_1_above[i]/flux_an_above_1_list[i]) for i in range(len(flux_an_above_1_list))]
rel_er_num_v_an_1_below = [abs(er_num_v_an_1_below[i]/flux_an_below_1_list[i])for i in range(len(flux_an_below_1_list))]
rel_er_num_v_full_1_above =  [abs(er_num_v_full_1_above[i]/flux_full_above_1_list[i]) for i in range(len(flux_full_above_1_list))]
rel_er_num_v_full_1_below =  [abs(er_num_v_full_1_below[i]/flux_full_below_1_list[i]) for i in range(len(flux_full_below_1_list))]
rel_er_an_v_full_1_above = [abs(er_an_v_full_1_above[i]/flux_full_above_1_list[i]) for i in range(len(flux_full_above_1_list))]
rel_er_an_v_full_1_below = [abs(er_an_v_full_1_below[i]/flux_full_below_1_list[i]) for i in range(len(flux_full_below_1_list))]
rel_er_num_v_an_2_above = [abs(er_num_v_an_2_above[i]/flux_an_above_2_list[i]) for i in range(len(flux_an_above_2_list))]
rel_er_num_v_an_2_below = [abs(er_num_v_an_2_below[i]/flux_an_below_2_list[i]) for i in range(len(flux_an_below_2_list))]
rel_er_num_v_full_2_above =  [abs(er_num_v_full_2_above[i]/flux_full_above_2_list[i]) for i in range(len(flux_full_above_2_list))]
rel_er_num_v_full_2_below =  [abs(er_num_v_full_2_below[i]/flux_full_below_2_list[i]) for i in range(len(flux_full_below_2_list))]
rel_er_an_v_full_2_above = [abs(er_an_v_full_2_above[i]/flux_full_above_2_list[i]) for i in range(len(flux_full_above_2_list))]
rel_er_an_v_full_2_below = [abs(er_an_v_full_2_below[i]/flux_full_below_2_list[i]) for i in range(len(flux_full_below_2_list))]

##plotting part


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


# for a v lambda

fig, ax = plt.subplots(1,1,figsize=(6, 4)) # sets figure size

ax.plot(exactplist, exact_flux_list, label='exact effective flux', linestyle='--', color='black') #plots on main axis
#ax.plot(permeability_list, av_flux_num_above_1_list, label='effective interface simulation', linestyle = '', marker = 's',markersize=10, color='turquoise') 
ax.plot(permeability_list, flux_full_above_1_list, label='full simulation', linestyle = '', marker = 'x',markersize=10, color='indigo') 
ax.legend(fontsize=10)

ax.set_xlabel(r'$P_{\text{eff}}$')
ax.set_ylabel(r'Average Flux')
ax.grid(True, linestyle='--', alpha=0.6)
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
ax.yaxis.get_major_formatter().set_scientific(False)
ax.yaxis.get_major_formatter().set_useOffset(False)


fig.savefig('steadyfluxvperm.eps',bbox_inches='tight',dpi=500) # saves as jpg with no whitespace. 

                                                                                                                    

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

fig, ax = plt.subplots(1,1,figsize=(6, 4)) # sets figure size

#ax.plot(permeability_list, rel_er_num_v_full_1_above, label='effective interface simulation vs full flux simulation', linestyle = '--', marker = 's', color='red') #plots on main axis
ax.plot(permeability_list, rel_er_full_v_exact_above, linestyle='dotted', marker='x', color='indigo') 
#ax.plot(permeability_list, rel_er_num_v_exact_above, label='effective interface simulation vs analytical', linestyle = '--', marker = 'o', color='turquoise') 
#ax.legend(fontsize=10)

ax.set_xlabel(r'$P_{\text{eff}}$')
ax.set_ylabel(r'Relative flux error')
ax.grid(True, linestyle='--', alpha=0.6)
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
ax.yaxis.get_major_formatter().set_scientific(False)
ax.yaxis.get_major_formatter().set_useOffset(False)


fig.savefig('steadyRelfluxerrorvperm.eps',bbox_inches='tight',dpi=500) # saves as jpg with no whitespace. 
