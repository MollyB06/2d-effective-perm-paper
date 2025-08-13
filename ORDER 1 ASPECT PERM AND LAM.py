########### ORDER 1 ASPECT RATIO PERMEABILITY AND LAMBDA VS ASPECT RATIO

from math import exp, pi, sqrt, log

from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
from scipy import integrate, interpolate


def integrand_1(x,lam):
    num = sqrt((abs(x-lam)*abs(x+ lam)))
    denom = sqrt(abs(x-1)*abs(x+1))
    return num/denom

def integrand_2(x,lam):
    num = sqrt(abs(x - (1/lam))*abs(x+(1/lam)))
    denom = sqrt(abs(x-1)*abs(x+1))
    return num/denom

lambda_small_list = np.linspace(2,20,10000)
a_small_list=[]
for lam in lambda_small_list:
    a_num= integrate.quad(integrand_1, 1, lam, args=(lam))[0] - lam*integrate.quad(integrand_2,(1/lam),1, args = (lam))[0]
    a_denom = 2*(integrate.quad(integrand_1,0,1,args=(lam))[0]+ lam*integrate.quad(integrand_2,0,(1/lam),args=(lam))[0])
    a_small_list.append(a_num/a_denom)


#####for small a lam = 1+ gam
gamma_list = [t/100000 for t in range(1,100000)]
lambda_tiny_list=[1+t/80000 for t in range(1,80000)]
a_tiny_list =[]
for lam in lambda_tiny_list:
    a_num= integrate.quad(integrand_1, 1, lam, args=(lam))[0] - lam*integrate.quad(integrand_2,(1/lam),1, args = (lam))[0]
    a_denom = 2*(integrate.quad(integrand_1,0,1,args=(lam))[0]+ lam*integrate.quad(integrand_2,0,(1/lam),args=(lam))[0])
    a_tiny_list.append(a_num/a_denom)

    
def Q(lam):
    denom = lam*integrate.quad(integrand_2,0,(1/lam),args=(lam))[0]+ integrate.quad(integrand_1,0,1,args=(lam))[0]
    return -1/denom



###set eps and delta values or loop over them

delta = 0.1
eps = 0.1

#### list of aspect ratios to check
aspect_list = a_tiny_list + a_small_list
lambdafull_list = list(lambda_tiny_list) + list(lambda_small_list)


Q_list =[Q(lam) for lam in lambdafull_list]
##full permeability
full_perm =[1/((delta/pi)*log(1/(4*eps**2*pi**2*Q_list[i]**2*lambdafull_list[i]))) for i in range(len(lambdafull_list))]
small_a_perm_lim =[1/((2*delta/pi)*log(1/(pi*eps))+(16/pi**2)*delta*a) for a in a_tiny_list]
large_a_perm_lim =[1/(a*delta+(2*delta/pi)*(log(1/(8*eps))+1)) for a in aspect_list]



#reduced range lists for inset plots
smaller_aspect_list = a_tiny_list
smaller_lambda_list = list(lambda_tiny_list)

smaller_Q_list =[Q(lam) for lam in smaller_lambda_list]
full_permsmall =[1/((delta/pi)*log(1/(4*eps**2*pi**2*smaller_Q_list[i]**2*smaller_lambda_list[i]))) for i in range(len(smaller_lambda_list))]
small_a_perm_v2 =[1/((2*delta/pi)*log(1/(pi*eps))+(16/pi**2)*delta*a) for a in smaller_aspect_list] 
large_a_perm_limsmall=[1/(a*delta+(2*delta/pi)*(log(1/(8*eps))+1)) for a in smaller_aspect_list]




#Plotting macros

plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
plt.rcParams.update({
    "text.usetex": True,  # Enable full LaTeX rendering
    "font.family": "serif",
    "font.serif": ["cmr10", "serif"],  # Optional, reinforces LaTeX look
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 30,
})


fig, ax = plt.subplots(1,1,figsize=(6, 4)) # sets figure size

ax.plot(aspect_list, full_perm, label='full solution', linestyle='solid', color='black') #plots on main axis
ax.plot(aspect_list, large_a_perm_lim, label= r'limit as $a \rightarrow \infty$', linestyle='dotted',  color='dodgerblue') 
ax.plot(a_tiny_list, small_a_perm_lim, label=r'limit as $a \rightarrow 0$', linestyle='dashed',  color='magenta') 
ax.legend(fontsize=15)
inset_ax= ax.inset_axes([.07, .095, .36, .27]) # [x, y, width, height] w.r.t. ax # setsup inset axes

inset_ax.plot(smaller_aspect_list, full_permsmall, label='full solution', linestyle='solid', color='black') # plots on inset
inset_ax.plot(smaller_aspect_list, large_a_perm_limsmall, label= r'limit as $a \rightarrow \infty$', linestyle='dotted',  color='dodgerblue') 
inset_ax.plot(smaller_aspect_list, small_a_perm_v2, label=r'limit as $a \rightarrow 0$', linestyle='dashed',  color='magenta')
#inset_ax.grid(True)

ax.set_xlabel(r'$a$')
ax.set_ylabel(r'$P_{\text{eff}}$')
ax.grid(True, linestyle='--', alpha=0.6)
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
ax.yaxis.get_major_formatter().set_scientific(False)
ax.yaxis.get_major_formatter().set_useOffset(False)


fig.savefig('aspectvperminset.eps',bbox_inches='tight',dpi=500) # saves as jpg with no whitespace. 




a_limits_small =[(lam-1)**2*(pi/16)-(lam-1)**3*(pi/16)+(lam-1)**4*(21*pi/512)-(lam-1)**4*pi/128*log(8/(lam-1)) for lam in lambda_tiny_list]
a_limits_largetiny =[(1/pi)*(log(4*lam)-2) for lam in lambda_tiny_list]
a_limits_small_leading =[(lam-1)**2*(pi/16) for lam in lambda_tiny_list]
a_limits_large =[(1/pi)*(log(4*lam)-2) for lam in lambdafull_list]


loglambda_full =[log(l) for l in lambdafull_list]
loglambdatiny =[log(l) for l in lambda_tiny_list]

# for a v lambda

fig, ax = plt.subplots(1,1,figsize=(6, 4)) # sets figure size

ax.plot(aspect_list, loglambda_full, label='full solution', linestyle='solid', color='black') #plots on main axis
ax.plot(a_limits_large, loglambda_full, label= r'limit as $a \rightarrow \infty$', linestyle='dotted',  color='dodgerblue') 
ax.plot(a_limits_small_leading, loglambdatiny, label=r'limit as $a \rightarrow 0$', linestyle='dashed',  color='magenta') 
ax.legend(fontsize=15)
inset_ax= ax.inset_axes([.58, .18, .38, .28]) # [x, y, width, height] w.r.t. ax # setsup inset axes

inset_ax.plot(a_tiny_list, loglambdatiny, label='full solution', linestyle='solid', color='black') # plots on inset
inset_ax.plot(a_limits_largetiny, loglambdatiny, label= r'limit as $a \rightarrow \infty$$', linestyle='dotted',  color='dodgerblue') 
inset_ax.plot(a_limits_small_leading, loglambdatiny, label=r'limit as $a \rightarrow 0$$ ', linestyle='dashed',  color='magenta')
ax.set_xlabel(r'$a$')
ax.set_ylabel(r'$\log{\lambda}$')
ax.grid(True, linestyle='--', alpha=0.6)
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
ax.yaxis.get_major_formatter().set_scientific(False)
ax.yaxis.get_major_formatter().set_useOffset(False)


fig.savefig('aspectvlaminset.eps',bbox_inches='tight',dpi=500) # saves as jpg with no whitespace. 


























