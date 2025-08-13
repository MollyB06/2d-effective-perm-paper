# 2d-effective-perm-paper
#All simulations and plotting code for paper: "Effective permeability conditions for diffusive transport through impermeable membranes with gaps" 

Authors: Molly Brennan contact: molly.brennan.22@ucl.ac.uk

Dependencies:
Version used: Python 2.12.2, FEniCSx

#Computational scripts
STEADY SIMULATIONS V2.py #runs all effective and full steady simulations for varying effective permeabilties varying the ratio of channel width to spacing, generates colourmaps (Fig 7 and 8) and perm v flux data (Fig 9(a)) and flux error v perm #data (Supplementary Fig 1(a))
TIME DEPENDENT SIMS V3.py #runs all time dependent simulations for 2 different effective permeabilties (P=0.189, P=0.953) generating flux v time data (Fig 9(b)) and flux error v time data (Supplementary Fig 1(b))
MESH SIZE CONVERGENCE V2.py # runs effective and full simulations for varying mesh size to generate mesh size v error data (Supplementary Fig 2(a))
TIME STEP CONVERGENCE V2.py #runs time dependent effective and full simulations for varying time step size to generate time step v error data (Supplementary Fig 2(b))


#plotting scripts
ORDER 1 ASPECT PERM AND LAM.py #generates aspect ratio v log(lambda) and aspect ratio v effective permeability plots (Fig 6)
STEADY PLOTTING.py #takes data from STEADY SIMULATIONS V2.py and generates flux v perm plot (Fig 9(a)) and flux error v perm (Supplementary Fig 1(a))
TIME DEP PLOTTING.py #takes data from TIME DEPENDENT SIMS V3.py and generates flux vs time plot (Fig 9(b)) and flux error v time plot (Supplementary Fig 1(b))  
MESH CONVERGENCE PLOTTER.py #takes data from MESH SIZE CONVERGNECE V2.py and generates mesh size v relative error plot (Supplementary Fig 2(a))
TIME STEP CON PLOTTING.py #takes data from TIME STEP CONVERGENCE V2.py and generates time step size v relative error plot (Supplementary Fig 2(b))



