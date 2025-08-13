#### code to compute and plot mesh size convergence for the steady simulation



import gmsh
import meshio
import numpy as np
import matplotlib.pyplot as plt
from petsc4py import PETSc
from dolfinx import default_scalar_type, io, geometry
from dolfinx.fem import (dirichletbc, Function, functionspace, locate_dofs_topological)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile
from dolfinx.io.gmshio import model_to_mesh
from dolfinx.plot import vtk_mesh
from dolfinx.fem import (Constant, Function, functionspace, assemble_scalar, dirichletbc, form, locate_dofs_topological, set_bc)
from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector, set_bc)
from dolfinx.mesh import create_mesh
from ufl import (FacetNormal, Measure, TestFunction, TrialFunction, TrialFunctions, TestFunctions, avg, dot, dx, inner, grad)
from ufl import  SpatialCoordinate
from mpi4py import MPI
import gmsh
import numpy as np
import pyvista
import ufl




comm = MPI.COMM_WORLD

# Define the desired mesh size
mesh_size_list = [0.1,0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01,0.009,0.008,0.007]



#empty lists to store flux and error values
er_num_v_an_1_above=[]
er_num_v_an_1_below=[]
er_num_v_full_1_above=[]
er_num_v_full_1_below=[]
er_an_v_full_1_above=[]
er_an_v_full_1_below=[]
er_num_v_an_2_above=[]
er_num_v_an_2_below=[]
er_num_v_full_2_above=[]
er_num_v_full_2_below=[]
er_an_v_full_2_above=[]
er_an_v_full_2_below=[]
flux_an_above_1_list = []
flux_an_below_1_list=[]
flux_full_above_1_list=[]
flux_full_below_1_list=[]
flux_full_squish_above_1_list=[]
flux_full_squish_below_1_list=[]
flux_num_above_1_list=[]
flux_num_below_1_list=[]
flux_an_above_2_list = []
flux_an_below_2_list=[]
flux_full_above_2_list=[]
flux_full_below_2_list=[]
flux_full_squish_above_2_list=[]
flux_full_squish_below_2_list=[]
flux_num_above_2_list=[]
flux_num_below_2_list=[]

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

def homfixedmesh(mesh_size):
    gdim = 2
    model_rank = 0

    #####################################################################EFFECTIVE PROBLEM######################################
    #mesh generation
    gmsh.initialize()

    proc = MPI.COMM_WORLD.rank
    #define markers for domains
    main_marker = 2
    interface_marker = 3
    top_wall_marker=4
    bottom_wall_marker=5
    bottom_sub_marker= 6
    top_sub_marker = 7

    flux_line_above_1_marker=10
    flux_line_below_1_marker=11
    flux_line_above_2_marker=12
    flux_line_below_2_marker=13



    if proc == 0:
        # We create one rectangle for each subdomain
        rec1 = gmsh.model.occ.addRectangle(0, 0, 0, width, R-L, tag=1)
        rec2 = gmsh.model.occ.addRectangle(0, R-L, 0, width, R-L, tag=2)
        # We fuse the two rectangles and keep the interface between them 
        gmsh.model.occ.fragment([(2, rec1)], [(2, rec2)])
        gmsh.model.occ.synchronize()
        

        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)
        # Add points for horizontal lines to calculate flux over
        p1_above = gmsh.model.occ.addPoint(0, R - L + dist_1, 0)
        p2_above = gmsh.model.occ.addPoint(width, R - L + dist_1, 0)
        line_above = gmsh.model.occ.addLine(p1_above, p2_above)
        gmsh.model.occ.synchronize()

        p1_above2 = gmsh.model.occ.addPoint(0, R - L + dist_2, 0)
        p2_above2= gmsh.model.occ.addPoint(width, R - L + dist_2, 0)
        line_above2 = gmsh.model.occ.addLine(p1_above2, p2_above2)
        gmsh.model.occ.synchronize()

        p1_below = gmsh.model.occ.addPoint(0, R - L - dist_1, 0)
        p2_below = gmsh.model.occ.addPoint(width, R - L - dist_1, 0)
        line_below = gmsh.model.occ.addLine(p1_below, p2_below)
        gmsh.model.occ.synchronize()

        p1_below2 = gmsh.model.occ.addPoint(0, R - L - dist_2, 0)
        p2_below2= gmsh.model.occ.addPoint(width, R - L - dist_2, 0)
        line_below2 = gmsh.model.occ.addLine(p1_below2, p2_below2)
        gmsh.model.occ.synchronize()


        gmsh.model.occ.synchronize()
               
        
        # Fragment the added lines with the existing geometry
        gmsh.model.occ.fragment([(1, line_above), (1, line_below), (1,line_above2),(1,line_below2)], gmsh.model.getEntities(dim=2))
        
       

        # Synchronize and mesh
        gmsh.model.occ.synchronize()
        

        #set mesh size for all points
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)

        #mark different boundaries
        top_wall=[]
        bottom_wall=[]
        side_walls=[]
        interface = []
   
        flux_line_above_1 =[]
        flux_line_below_1 =[]
        flux_line_above_2 =[]
        flux_line_below_2 =[]
        for line in gmsh.model.getEntities(dim=1):
            com = gmsh.model.occ.getCenterOfMass(line[0], line[1])
            if np.isclose(com[1],0):
                bottom_wall.append(line[1])
            elif np.isclose(com[1],2*R-2*L):
                top_wall.append(line[1])
            elif np.isclose(com[1],R-L):
                interface.append(line[1])
            elif np.isclose(com[1], R-L-(dist_1/2)):
                side_walls.append(line[1])
            elif np.isclose(com[1], R-L+dist_1/2):
                side_walls.append(line[1])           
            elif np.isclose(com[1],R-L-dist_1 -(dist_2-dist_1)/2):
                side_walls.append(line[1])
            elif np.isclose(com[1],R-L+dist_1+(dist_2-dist_1)/2):
                side_walls.append(line[1])
            elif np.isclose(com[1],(R-L-dist_2)/2):
                side_walls.append(line[1])
            elif np.isclose(com[1], (3*(R-L)+dist_2)/2):
                side_walls.append(line[1])
            elif np.isclose(com[1],R-L-dist_1):
                flux_line_below_1.append(line[1])
            elif np.isclose(com[1],R-L + dist_1):
                flux_line_above_1.append(line[1])
            elif np.isclose(com[1],R-L-dist_2):
                flux_line_below_2.append(line[1])
            elif np.isclose(com[1],R-L + dist_2):
                flux_line_above_2.append(line[1])



       #mark subdomains
        
        
        bottom_half= []
        top_half =[]
        for surface in gmsh.model.getEntities(dim=2):
            com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
            if np.allclose(com, [width/2, (R-L-dist_2)/2, 0]):
                bottom_half.append(surface[1])
            elif np.allclose(com,[width/2, R-L - dist_1 - (dist_2-dist_1)/2,0]):
                bottom_half.append(surface[1])
            elif np.allclose(com,[width/2, R-L - dist_1/2,0]):
                bottom_half.append(surface[1])
            elif np.allclose(com, [width/2, R-L +dist_1/2,0]):
                top_half.append(surface[1])
            elif np.allclose(com, [width/2, R-L+dist_1+(dist_2-dist_1)/2, 0]):
                top_half.append(surface[1]) 
            elif np.allclose(com, [width/2, (3*(R-L)+dist_2)/2,0]):
                top_half.append(surface[1])
        
        # Synchronize and mesh
        gmsh.model.occ.synchronize()   
            

        gmsh.model.addPhysicalGroup(2, top_half, top_sub_marker)
        gmsh.model.addPhysicalGroup(2, bottom_half, bottom_sub_marker)
        gmsh.model.addPhysicalGroup(1, top_wall, top_wall_marker)
        gmsh.model.addPhysicalGroup(1, bottom_wall, bottom_wall_marker)
        gmsh.model.addPhysicalGroup(1, interface, interface_marker)
        gmsh.model.addPhysicalGroup(1, flux_line_above_1, flux_line_above_1_marker)
        gmsh.model.addPhysicalGroup(1, flux_line_below_1, flux_line_below_1_marker)
        gmsh.model.addPhysicalGroup(1, flux_line_above_2, flux_line_above_2_marker)
        gmsh.model.addPhysicalGroup(1, flux_line_below_2, flux_line_below_2_marker)
        # Synchronize and mesh
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.Algorithm", 2) #algorithms 2 (automatic) or 7 (BAMG(useful when certain directions need finer refinement))
        gmsh.model.mesh.generate(gdim)
        gmsh.model.mesh.optimize("Netgen")
        gmsh.write("mesh.msh")


    gmsh.finalize()

    

    #Convert the Gmsh mesh to Meshio and then to FEniCSx format
    def create_mesh(mesh, cell_type, prune_z=False): #cell type gives type of cells to extract, e.g. traingles for 2D problems
        cells = mesh.get_cells_type(cell_type)
        cell_data = mesh.get_cell_data("gmsh:physical", cell_type) #retrieves phyiscal markers for the cells
        points = mesh.points[:, :2] if prune_z else mesh.points
        out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={"name_to_read": [cell_data.astype(np.int32)]})
        return out_mesh

    if proc == 0:
        # Read in mesh
        msh = meshio.read("mesh.msh")

        
        MPI.COMM_WORLD.barrier()

        # Create and save one file for the mesh, and one file for the facets
        triangle_mesh = create_mesh(msh, "triangle", prune_z=True)
        line_mesh = create_mesh(msh, "line", prune_z=True)
        meshio.write("mesh.xdmf", triangle_mesh)
        meshio.write("mt.xdmf", line_mesh)

    MPI.COMM_WORLD.barrier()
    meshio_mesh = meshio.read("mt.xdmf")

    with io.XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "r") as xdmf:
        mesh_hom = xdmf.read_mesh(name="Grid")
        ct_hom = xdmf.read_meshtags(mesh_hom, name="Grid")

    mesh_hom.topology.create_connectivity(mesh_hom.topology.dim, mesh_hom.topology.dim - 1)


    with io.XDMFFile(MPI.COMM_WORLD, "mt.xdmf", "r") as xdmf:
        ft_hom = xdmf.read_meshtags(mesh_hom, name="Grid")


    tdim = mesh_hom.topology.dim #topological dimension of domain

    

    #define integration measures
    dx_hom = Measure("dx", domain = mesh_hom, subdomain_data= ct_hom)
    dS_hom = Measure("dS", mesh_hom, subdomain_data = ft_hom) 


    #MIXED ELEMENT FORMULATION
    #define function space and mixed elements
    V1 = functionspace(mesh_hom, ("CG", 1))
    V2 = functionspace(mesh_hom, ("CG", 1))

    #define mixed function space
    V_hom = functionspace(mesh_hom, V1.ufl_element() * V2.ufl_element())


    #define trial and test functions
    (u_1,u_2)= TrialFunctions(V_hom)
    (w_1,w_2)= TestFunctions(V_hom)

    #interface condition constant
    p = Constant(mesh_hom, PETSc.ScalarType(P))



    #MIXED FUNCTION SPACE BOUNDARY CONDITIONS AND ASSEMBLY
    #define boundary conditionS
    outer_below_bc = dirichletbc(default_scalar_type(1),locate_dofs_topological(V_hom.sub(0), tdim-1, ft_hom.find(bottom_wall_marker)),V_hom.sub(0))
    outer_above_bc = dirichletbc(default_scalar_type(0), locate_dofs_topological(V_hom.sub(1), tdim-1, ft_hom.find(top_wall_marker)),V_hom.sub(1))

    bc = [outer_below_bc, outer_above_bc]

    #jump condition to be applied across interface
    jmp_u = avg(u_2) - avg(u_1)
    jmp_w = avg(w_2)- avg(w_1)

    

    #linear and bilieanr forms
    a_hom= inner(grad(u_1), grad(w_1))*dx_hom(bottom_sub_marker)+ inner(grad(u_2), grad(w_2))*dx_hom(top_sub_marker) + p*jmp_u*jmp_w*dS_hom(interface_marker)
    l_hom= Constant(mesh_hom, default_scalar_type(0))*w_1*dx_hom(bottom_sub_marker) + Constant(mesh_hom, default_scalar_type(0))*w_2*dx_hom(top_sub_marker)

    

    #matrix assembly with boundary conditions
    A_hom = assemble_matrix(form(a_hom), bc)
    A_hom.assemble()
    b_hom= assemble_vector(form(l_hom))


    apply_lifting(b_hom, [form(a_hom)], [bc])
    b_hom.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b_hom, bc)

    A_hom.assemble()
    

    #solve
    w_sol = Function(V_hom)
    solver = PETSc.KSP().create(comm)
    # Set the solver type to 'preonly' (direct solve)
    solver.setType(PETSc.KSP.Type.PREONLY)
    # Set the preconditioner type to 'lu' (direct factorization)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.LU)
    solver.setOperators(A_hom)

    A_hom.shift(1e-10) # small shift solves invertibility issues
    solver.solve(b_hom, w_sol.vector)

    u_sol, v_sol = w_sol.split()



    



    #interpolate to visualise
    V_vis = functionspace(mesh_hom, ("CG",1))
    u_1_vis = Function(V_vis)
    u_2_vis = Function(V_vis)
    u_1_vis.interpolate(u_sol)
    u_2_vis.interpolate(v_sol)
    u_vis = Function(V_vis)


    u1_vis_array = u_1_vis.x.array
    u2_vis_array =u_2_vis.x.array
    
    coords = mesh_hom.geometry.x
    y_coords = coords[:, 1]  # y-coordinate of the mesh vertices

    # Define a threshold to identify the upper half of the domain
    threshold = R - L - epsilon

    # Find indices where y-coordinates are greater than the threshold
    indices1 = np.where(y_coords > threshold)[0]
    indices2 = np.where(y_coords < threshold)[0]

    # Set the corresponding solution values to zero so solutions only exist in their corresponding half of the domain
    u1_vis_array[indices1] = 0
    u2_vis_array[indices2] = 0

    u_vis.x.array[:] = u1_vis_array + u2_vis_array



    

    ########################################################### ANALYTICAL SOLUTION #######################################################
    #interface condition constant
    p = Constant(mesh_hom, PETSc.ScalarType(P))

    x = SpatialCoordinate(mesh_hom)

    #define analytical solution

    def u1(x):
        return 1 - (P*x[1])/(1 + 2*P*(R-L))

    def u2(x):
        return  (P/(1 + 2*P*(R-L)))*(2*R-2*L - x[1])


    # project and plot
    V_an = functionspace(mesh_hom, ("CG",1))
    u1d = Function(V_an)
    u1d.interpolate(u1)
    u2d = Function(V_an)
    u2d.interpolate(u2)
    u_an = Function(V_an)


    u1_solution_array = u1d.x.array
    u2_solution_array =u2d.x.array

    # Manipulate values so that function is zero where it shouldn't be defined
    coords = mesh_hom.geometry.x
    y_coords = coords[:, 1]  # y-coordinate of the mesh vertices

    # Define a threshold to identify the upper half of the domain
    threshold = R - L - epsilon

    # Find indices where y-coordinates are greater than the threshold
    indices1 = np.where(y_coords > threshold)[0]
    indices2 = np.where(y_coords < threshold)[0]

    # Set the corresponding solution values to zero
    u1_solution_array[indices1] = 0
    u2_solution_array[indices2] = 0

    u_an.vector[:] = u1_solution_array + u2_solution_array



    


### FLUX CALCULATIONS
    


    #NUMERICAL SOLUTION OF EFFECTIVE
    
    flux = -grad(u_vis)

    # Restrict flux to one side of the facet
    flux_normal_hom = avg(flux[1])


    # Define the integral using the restricted flux
    integral_flux_above_1_hom = flux_normal_hom * dS_hom(flux_line_above_1_marker)
    integral_flux_below_1_hom = flux_normal_hom * dS_hom(flux_line_below_1_marker)
    integral_flux_above_2_hom = flux_normal_hom * dS_hom(flux_line_above_2_marker)
    integral_flux_below_2_hom = flux_normal_hom * dS_hom(flux_line_below_2_marker)

    # Assemble the integral
    flux_total_above_1_hom = assemble_scalar(form(integral_flux_above_1_hom))
    flux_total_below_1_hom = assemble_scalar(form(integral_flux_below_1_hom))
    flux_total_above_2_hom = assemble_scalar(form(integral_flux_above_2_hom))
    flux_total_below_2_hom = assemble_scalar(form(integral_flux_below_2_hom))


    # Average flux over the line
    line_length_1_above = assemble_scalar(form(1 * dS_hom(flux_line_above_1_marker)))
    line_length_1_below = assemble_scalar(form(1 * dS_hom(flux_line_below_1_marker)))
    line_length_2_above = assemble_scalar(form(1 * dS_hom(flux_line_above_2_marker)))
    line_length_2_below = assemble_scalar(form(1 * dS_hom(flux_line_below_2_marker)))
    average_flux_above_1_hom = flux_total_above_1_hom / line_length_1_above
    average_flux_below_1_hom = flux_total_below_1_hom / line_length_1_below
    average_flux_above_2_hom = flux_total_above_2_hom / line_length_2_above
    average_flux_below_2_hom = flux_total_below_2_hom / line_length_2_below
    flux_num_above_1_list.append(average_flux_above_1_hom)
    flux_num_below_1_list.append(average_flux_below_1_hom)
    flux_num_above_2_list.append(average_flux_above_2_hom)
    flux_num_below_2_list.append(average_flux_below_2_hom)



    #ANALYTICAL SOLUTION OF EFFECTIVE
    # Assuming u_an is a finite element function
    flux_an = -grad(u_an)

    flux_normal_an = avg(flux_an[1])

    # Define the integral using the restricted flux
    integral_flux_above_1_an = flux_normal_an * dS_hom(flux_line_above_1_marker)
    integral_flux_below_1_an = flux_normal_an*dS_hom(flux_line_below_1_marker)
    integral_flux_above_2_an = flux_normal_an * dS_hom(flux_line_above_2_marker)
    integral_flux_below_2_an = flux_normal_an*dS_hom(flux_line_below_2_marker)
    
    # Assemble the integral
    flux_total_above_1_an = assemble_scalar(form(integral_flux_above_1_an))
    flux_total_below_1_an = assemble_scalar(form(integral_flux_below_1_an))
    flux_total_above_2_an = assemble_scalar(form(integral_flux_above_2_an))
    flux_total_below_2_an = assemble_scalar(form(integral_flux_below_2_an))
    

    





    # Average flux over the line
    average_flux_above_1_an = flux_total_above_1_an / line_length_1_above
    average_flux_below_1_an = flux_total_below_1_an/line_length_1_below
    average_flux_above_2_an = flux_total_above_2_an / line_length_2_above
    average_flux_below_2_an = flux_total_below_2_an/line_length_2_below
    flux_an_above_1_list.append(average_flux_above_1_an)
    flux_an_below_1_list.append(average_flux_below_1_an)
    flux_an_above_2_list.append(average_flux_above_2_an)
    flux_an_below_2_list.append(average_flux_below_2_an)



def fullfixedmesh(mesh_size):
    gdim = 2
    model_rank = 0

    ####################effective mesh for flux calculations
    #mesh generation
    gmsh.initialize()

    proc = MPI.COMM_WORLD.rank
    #define markers for domains
    main_marker = 2
    interface_marker = 3
    top_wall_marker=4
    bottom_wall_marker=5
    bottom_sub_marker= 6
    top_sub_marker = 7

    flux_line_above_1_marker=10
    flux_line_below_1_marker=11
    flux_line_above_2_marker=12
    flux_line_below_2_marker=13



    if proc == 0:
        # We create one rectangle for each subdomain
        rec1 = gmsh.model.occ.addRectangle(0, 0, 0, width, R-L, tag=1)
        rec2 = gmsh.model.occ.addRectangle(0, R-L, 0, width, R-L, tag=2)
        # We fuse the two rectangles and keep the interface between them 
        gmsh.model.occ.fragment([(2, rec1)], [(2, rec2)])
        gmsh.model.occ.synchronize()
        

        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)
        # Add points for horizontal lines to calculate flux over
        p1_above = gmsh.model.occ.addPoint(0, R - L + dist_1, 0)
        p2_above = gmsh.model.occ.addPoint(width, R - L + dist_1, 0)
        line_above = gmsh.model.occ.addLine(p1_above, p2_above)
        gmsh.model.occ.synchronize()

        p1_above2 = gmsh.model.occ.addPoint(0, R - L + dist_2, 0)
        p2_above2= gmsh.model.occ.addPoint(width, R - L + dist_2, 0)
        line_above2 = gmsh.model.occ.addLine(p1_above2, p2_above2)
        gmsh.model.occ.synchronize()

        p1_below = gmsh.model.occ.addPoint(0, R - L - dist_1, 0)
        p2_below = gmsh.model.occ.addPoint(width, R - L - dist_1, 0)
        line_below = gmsh.model.occ.addLine(p1_below, p2_below)
        gmsh.model.occ.synchronize()

        p1_below2 = gmsh.model.occ.addPoint(0, R - L - dist_2, 0)
        p2_below2= gmsh.model.occ.addPoint(width, R - L - dist_2, 0)
        line_below2 = gmsh.model.occ.addLine(p1_below2, p2_below2)
        gmsh.model.occ.synchronize()


        gmsh.model.occ.synchronize()
               
        
        # Fragment the added lines with the existing geometry
        gmsh.model.occ.fragment([(1, line_above), (1, line_below), (1,line_above2),(1,line_below2)], gmsh.model.getEntities(dim=2))
        
       

        # Synchronize and mesh
        gmsh.model.occ.synchronize()
        

        #set mesh size for all points
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)

        #mark different boundaries
        top_wall=[]
        bottom_wall=[]
        side_walls=[]
        interface = []
   
        flux_line_above_1 =[]
        flux_line_below_1 =[]
        flux_line_above_2 =[]
        flux_line_below_2 =[]
        for line in gmsh.model.getEntities(dim=1):
            com = gmsh.model.occ.getCenterOfMass(line[0], line[1])
            if np.isclose(com[1],0):
                bottom_wall.append(line[1])
            elif np.isclose(com[1],2*R-2*L):
                top_wall.append(line[1])
            elif np.isclose(com[1],R-L):
                interface.append(line[1])
            elif np.isclose(com[1], R-L-(dist_1/2)):
                side_walls.append(line[1])
            elif np.isclose(com[1], R-L+dist_1/2):
                side_walls.append(line[1])           
            elif np.isclose(com[1],R-L-dist_1 -(dist_2-dist_1)/2):
                side_walls.append(line[1])
            elif np.isclose(com[1],R-L+dist_1+(dist_2-dist_1)/2):
                side_walls.append(line[1])
            elif np.isclose(com[1],(R-L-dist_2)/2):
                side_walls.append(line[1])
            elif np.isclose(com[1], (3*(R-L)+dist_2)/2):
                side_walls.append(line[1])
            elif np.isclose(com[1],R-L-dist_1):
                flux_line_below_1.append(line[1])
            elif np.isclose(com[1],R-L + dist_1):
                flux_line_above_1.append(line[1])
            elif np.isclose(com[1],R-L-dist_2):
                flux_line_below_2.append(line[1])
            elif np.isclose(com[1],R-L + dist_2):
                flux_line_above_2.append(line[1])



       #mark subdomains
        
        
        bottom_half= []
        top_half =[]
        for surface in gmsh.model.getEntities(dim=2):
            com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
            if np.allclose(com, [width/2, (R-L-dist_2)/2, 0]):
                bottom_half.append(surface[1])
            elif np.allclose(com,[width/2, R-L - dist_1 - (dist_2-dist_1)/2,0]):
                bottom_half.append(surface[1])
            elif np.allclose(com,[width/2, R-L - dist_1/2,0]):
                bottom_half.append(surface[1])
            elif np.allclose(com, [width/2, R-L +dist_1/2,0]):
                top_half.append(surface[1])
            elif np.allclose(com, [width/2, R-L+dist_1+(dist_2-dist_1)/2, 0]):
                top_half.append(surface[1]) 
            elif np.allclose(com, [width/2, (3*(R-L)+dist_2)/2,0]):
                top_half.append(surface[1])
        
        # Synchronize and mesh
        gmsh.model.occ.synchronize()   
            

        gmsh.model.addPhysicalGroup(2, top_half, top_sub_marker)
        gmsh.model.addPhysicalGroup(2, bottom_half, bottom_sub_marker)
        gmsh.model.addPhysicalGroup(1, top_wall, top_wall_marker)
        gmsh.model.addPhysicalGroup(1, bottom_wall, bottom_wall_marker)
        gmsh.model.addPhysicalGroup(1, interface, interface_marker)
        gmsh.model.addPhysicalGroup(1, flux_line_above_1, flux_line_above_1_marker)
        gmsh.model.addPhysicalGroup(1, flux_line_below_1, flux_line_below_1_marker)
        gmsh.model.addPhysicalGroup(1, flux_line_above_2, flux_line_above_2_marker)
        gmsh.model.addPhysicalGroup(1, flux_line_below_2, flux_line_below_2_marker)
        # Synchronize and mesh
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.Algorithm", 2) #algorithms 2 (automatic) or 7 (BAMG(useful when certain directions need finer refinement))
        gmsh.model.mesh.generate(gdim)
        gmsh.model.mesh.optimize("Netgen")
        gmsh.write("mesh.msh")


    gmsh.finalize()

    

    #Convert the Gmsh mesh to Meshio and then to FEniCSx format
    def create_mesh(mesh, cell_type, prune_z=False): #cell type gives type of cells to extract, e.g. traingles for 2D problems
        cells = mesh.get_cells_type(cell_type)
        cell_data = mesh.get_cell_data("gmsh:physical", cell_type) #retrieves phyiscal markers for the cells
        points = mesh.points[:, :2] if prune_z else mesh.points
        out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={"name_to_read": [cell_data.astype(np.int32)]})
        return out_mesh

    if proc == 0:
        # Read in mesh
        msh = meshio.read("mesh.msh")

        
        MPI.COMM_WORLD.barrier()

        # Create and save one file for the mesh, and one file for the facets
        triangle_mesh = create_mesh(msh, "triangle", prune_z=True)
        line_mesh = create_mesh(msh, "line", prune_z=True)
        meshio.write("mesh.xdmf", triangle_mesh)
        meshio.write("mt.xdmf", line_mesh)

    MPI.COMM_WORLD.barrier()
    meshio_mesh = meshio.read("mt.xdmf")

    with io.XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "r") as xdmf:
        mesh_hom = xdmf.read_mesh(name="Grid")
        ct_hom = xdmf.read_meshtags(mesh_hom, name="Grid")

    mesh_hom.topology.create_connectivity(mesh_hom.topology.dim, mesh_hom.topology.dim - 1)


    with io.XDMFFile(MPI.COMM_WORLD, "mt.xdmf", "r") as xdmf:
        ft_hom = xdmf.read_meshtags(mesh_hom, name="Grid")


    tdim = mesh_hom.topology.dim #topological dimension of domain

    

    #define integration measures
    dx_hom = Measure("dx", domain = mesh_hom, subdomain_data= ct_hom)
    dS_hom = Measure("dS", mesh_hom, subdomain_data = ft_hom) 


   
    #interpolate to visualise
    V_vis = functionspace(mesh_hom, ("CG",1))


    ##################################### FULL SIMULATION ###########################################################################

    gmsh.initialize()

    #defining where impermeable chunks will go

    membrane_corners_x = [delta*eps + delta/2 + i*delta for i in range(0,n-1)]
    c_y = R-L
    normal_chunk_width = (delta - 2*eps*delta)
    end_chunk_width = normal_chunk_width/2
    c_x_final = width - end_chunk_width
    mass_of_mem = (delta - 2*delta*eps)*2*L
    com_start = np.array([(delta- 2*delta*eps)/4,R,0])
    com_end = np.array([(width - (delta - 2* delta*eps)/4),R,0])


    membrane_com = []
    for corner in membrane_corners_x:
        new = tuple(a+b for a,b in zip((end_chunk_width,R,0),(corner,0,0)))
        membrane_com.append(new)
    membrane_com.append(((delta- 2*delta*eps)/4,R,0))
    membrane_com.append(((width - (delta - 2* delta*eps)/4),R,0))



    gdim = 2
    mesh_comm = MPI.COMM_WORLD
    model_rank = 0
    if mesh_comm.rank == model_rank:
        rectangle = gmsh.model.occ.addRectangle(0, 0, 0, width, 2*R, tag=1)
        membrane_start = gmsh.model.occ.addRectangle(0,c_y,0, end_chunk_width, 2*L)
        membrane_end= gmsh.model.occ.addRectangle(c_x_final,c_y,0, end_chunk_width, 2*L)
        membrane_bits = [(2,gmsh.model.occ.addRectangle(c_x, c_y, 0, normal_chunk_width, 2*L))for c_x in membrane_corners_x]
        
        membrane= [(2,membrane_start),(2,membrane_end)]
        membrane.extend(membrane_bits)

        
        whole_domain = gmsh.model.occ.cut([(gdim, rectangle)], membrane)
        gmsh.model.occ.synchronize()

        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)
        # Add points for horizontal lines to calculate flux over
        p1_above = gmsh.model.occ.addPoint(0, R + L + dist_1, 0)
        p2_above = gmsh.model.occ.addPoint(width, R + L + dist_1, 0)
        line_above = gmsh.model.occ.addLine(p1_above, p2_above)
        gmsh.model.occ.synchronize()

        p1_above2 = gmsh.model.occ.addPoint(0, R + L + dist_2, 0)
        p2_above2= gmsh.model.occ.addPoint(width, R + L + dist_2, 0)
        line_above2 = gmsh.model.occ.addLine(p1_above2, p2_above2)
        gmsh.model.occ.synchronize()

        p1_below = gmsh.model.occ.addPoint(0, R - L - dist_1, 0)
        p2_below = gmsh.model.occ.addPoint(width, R - L - dist_1, 0)
        line_below = gmsh.model.occ.addLine(p1_below, p2_below)
        gmsh.model.occ.synchronize()

        p1_below2 = gmsh.model.occ.addPoint(0, R - L - dist_2, 0)
        p2_below2= gmsh.model.occ.addPoint(width, R - L - dist_2, 0)
        line_below2 = gmsh.model.occ.addLine(p1_below2, p2_below2)
        gmsh.model.occ.synchronize()

   
        gmsh.model.occ.synchronize()
               
        
        # Fragment the added lines with the existing geometry
        gmsh.model.occ.fragment([(1, line_above), (1, line_below), (1,line_above2),(1,line_below2)], gmsh.model.getEntities(dim=2))
    
        
        # Synchronize and mesh
        gmsh.model.occ.synchronize()
        

        #set mesh size for all points
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)
        


        #tag boundaries
        top_wall=[]
        bottom_wall=[]
        side_walls=[]
        membrane_boundary=[]
        flux_line_above =[]
        flux_line_below =[]
        for line in gmsh.model.getEntities(dim=1):
            com = gmsh.model.occ.getCenterOfMass(line[0], line[1])
            if np.isclose(com[1],0):
                bottom_wall.append(line[1])
            elif np.isclose(com[1],2*R):
                top_wall.append(line[1])
            elif np.isclose(com[1], R-L-(dist_1/2)):
                side_walls.append(line[1])
            elif np.isclose(com[1], R+L+dist_1/2):
                side_walls.append(line[1])           
            elif np.isclose(com[1],R-L-dist_1 -(dist_2-dist_1)/2):
                side_walls.append(line[1])
            elif np.isclose(com[1],R+L+dist_1+(dist_2-dist_1)/2):
                side_walls.append(line[1])
            elif np.isclose(com[1],(R-L-dist_2)/2):
                side_walls.append(line[1])
            elif np.isclose(com[1], (3*R + L +dist_2)/2):
                side_walls.append(line[1])
            elif np.isclose(com[1],R-L-dist_1):
                flux_line_below_1.append(line[1])
            elif np.isclose(com[1],R+L + dist_1):
                flux_line_above_1.append(line[1])
            elif np.isclose(com[1],R-L-dist_2):
                flux_line_below_2.append(line[1])
            elif np.isclose(com[1],R+L + dist_2):
                flux_line_above_2.append(line[1])
            else:
                membrane_boundary.append(line[1])

        #add tag for domain
        domain_volume=[]
        for surface in gmsh.model.getEntities(dim=2):
            com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
            domain_volume.append(surface[1])


   
        
        #set tags
        top_wall_marker=1
        bottom_wall_marker=2
        side_walls_marker=3
        membrane_walls_marker=4




        #add membranes as physical groups in the mesh
        gmsh.model.addPhysicalGroup(2, domain_volume,0)
        gmsh.model.addPhysicalGroup(1, top_wall, top_wall_marker)
        gmsh.model.addPhysicalGroup(1, bottom_wall, bottom_wall_marker)
        gmsh.model.addPhysicalGroup(1, membrane_boundary, membrane_walls_marker)
        gmsh.model.addPhysicalGroup(1, flux_line_above_1, flux_line_above_1_marker)
        gmsh.model.addPhysicalGroup(1, flux_line_below_1, flux_line_below_1_marker)
        gmsh.model.addPhysicalGroup(1, flux_line_above_2, flux_line_above_2_marker)
        gmsh.model.addPhysicalGroup(1, flux_line_below_2, flux_line_below_2_marker)
        
        
        #meshing fields to define how mesh varies throughout the domain making it finer close to the membrane edges
        #creates distance field to calculate distance from certain edges
        gmsh.model.mesh.field.add("Distance",1)
        gmsh.model.mesh.field.setNumbers(1, "EdgesList", membrane_boundary)
        #adds threshold field to control the mesh size based on the distance calculated by distance field
        gmsh.model.mesh.field.add("Threshold",2)
        gmsh.model.mesh.field.setNumber(2,"IField",1) #links threshold field to distance field
        gmsh.model.mesh.field.setNumber(2,"LcMin", mesh_size/5) #sets minimum element size
        gmsh.model.mesh.field.setNumber(2,"LcMax",  5*mesh_size) #sets maximum element size
        gmsh.model.mesh.field.setNumber(2,"DistMin", 2*mesh_size) # sets the distance from the edges from which element size grows from minimum
        gmsh.model.mesh.field.setNumber(2, "DistMax", 20*mesh_size) # sets distance above which element size becomes maximum
        #use minimum field to ensure mesh size adheres to smallest size defined by threshold field
        gmsh.model.mesh.field.add("Min",5) #adds minumum field
        gmsh.model.mesh.field.setNumbers(5, "FieldsList", [2]) #sets minimum field to include the threshold field ID
        gmsh.model.mesh.field.setAsBackgroundMesh(5) #sets minimum field as background field making it the controlling field for mesh generation
        #use gmsh to generate the mesh using defined fields and physical groups defining the meshing algorithm
        gmsh.option.setNumber("Mesh.Algorithm", 2) #try algorithms 2 (automatic) or 7 (BAMG(useful when certain directions need finer refinement))
        gmsh.model.mesh.generate(gdim)
        gmsh.model.mesh.optimize("Netgen")




    #convert the gmsh model to fenics mesh

    mesh_full, ct, ft= model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=2) #sets the mesh, cell tags and facet tags
    ft.name = "Facet markers"

    gmsh.finalize()


    #full solution and fluxes
    V = functionspace(mesh_full, ("CG",1)) #define function space
    tdim = mesh_full.topology.dim #topological dimension of domain


    outer_below_bc = dirichletbc(default_scalar_type(1),locate_dofs_topological(V, tdim-1, ft.find(bottom_wall_marker)),V)
    outer_above_bc = dirichletbc(default_scalar_type(0), locate_dofs_topological(V, tdim-1, ft.find(top_wall_marker)),V)
    bcs = [outer_below_bc, outer_above_bc]

    u = TrialFunction(V)
    v = TestFunction(V)

    dx_full = Measure("dx", domain = mesh_full, subdomain_data= ct)

    a = dot(grad(u),grad(v))*dx
    L_full = Constant(mesh_full, default_scalar_type(0)) *v*dx_full

    problem = LinearProblem(a, L_full, bcs=bcs,petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()


### FLUX CALCULATIONS
    dS_full = Measure("dS", mesh_full, subdomain_data = ft) 
    # # Assuming u_vis is a finite element function
    flux_full = -grad(uh)



    # # Restrict flux to one side of the facet
    flux_normal_full = avg(flux_full[1])


    # Define the integral using the restricted flux
    integral_flux_above_1_full = flux_normal_full * dS_full(flux_line_above_1_marker)
    integral_flux_below_1_full = flux_normal_full * dS_full(flux_line_below_1_marker)
    integral_flux_above_2_full = flux_normal_full * dS_full(flux_line_above_2_marker)
    integral_flux_below_2_full = flux_normal_full * dS_full(flux_line_below_2_marker)

    # Assemble the integral
    flux_total_above_1_full = assemble_scalar(form(integral_flux_above_1_full))
    flux_total_below_1_full = assemble_scalar(form(integral_flux_below_1_full))
    flux_total_above_2_full = assemble_scalar(form(integral_flux_above_2_full))
    flux_total_below_2_full = assemble_scalar(form(integral_flux_below_2_full))
    


    # Average flux over the line
    line_length_above_1_full = assemble_scalar(form(1 * dS_full(flux_line_above_1_marker)))
    average_flux_above_1_full = flux_total_above_1_full / line_length_above_1_full
    line_length_below_1_full = assemble_scalar(form(1 * dS_full(flux_line_below_1_marker)))
    average_flux_below_1_full = flux_total_below_1_full / line_length_below_1_full
    line_length_above_2_full = assemble_scalar(form(1 * dS_full(flux_line_above_2_marker)))
    average_flux_above_2_full = flux_total_above_2_full / line_length_above_2_full
    line_length_below_2_full = assemble_scalar(form(1 * dS_full(flux_line_below_2_marker)))
    average_flux_below_2_full = flux_total_below_2_full / line_length_below_2_full

    flux_full_above_1_list.append(average_flux_above_1_full)
    flux_full_below_1_list.append(average_flux_below_1_full)
    flux_full_above_2_list.append(average_flux_above_2_full)
    flux_full_below_2_list.append(average_flux_below_2_full)


    ##INTERPOLATE FULL SOLUTIN ONTO MESH USED FOR EFFECTIVE PROBLEM FOR COMPARISONS
    coords_hom = mesh_hom.geometry.x
    values_full_squish = np.zeros(coords_hom.shape[0])

    u_full_squish = Function(V_vis)



    # Create bounding box tree for old mesh

    bb_tree_full = geometry.bb_tree(mesh_full,tdim)


    # Interpolate values
    for i, coord in enumerate(coords_hom):
        x, y, _ = coord  # Extract only x and y components, ignore the third (z) component if present
        if y <= R - L - epsilon:
            # Directly use the old function value
            point = np.array([x, y, 0.0])
            cell_candidates = geometry.compute_collisions_points(bb_tree_full, point)
            colliding_cells = geometry.compute_colliding_cells(mesh_full, cell_candidates, point)
            if len(colliding_cells) > 0:
                closest_cell = colliding_cells[0]
                values_full_squish[i] = uh.eval([point], [closest_cell])[0]
            else:
                values_full_squish[i] = 0
        elif y >= R - L- epsilon:
            # Shift y-coordinate by +2L
            shifted_coord = (x, y +2 * L, 0.0)
            cell_candidates = geometry.compute_collisions_points(bb_tree_full, shifted_coord)
            colliding_cells = geometry.compute_colliding_cells(mesh_full, cell_candidates, shifted_coord)
            if len(colliding_cells) > 0:
                closest_cell = colliding_cells[0]
                values_full_squish[i] = uh.eval([shifted_coord], [closest_cell])[0]
            else:
                values_full_squish[i] = 0
        else:
            # Set values to zero or some default if desired
            values_full_squish[i] = 0


    # Assign interpolated values to the new function
    u_full_squish.x.array[:] = values_full_squish


    # Assuming u_vis is a finite element function
    flux_full_squish = -grad(u_full_squish)



    # Restrict flux to one side of the facet
    flux_normal_squish = avg(flux_full_squish[1]) 

    # Define the integral using the restricted flux
    integral_flux_above_1_squish = flux_normal_squish * dS_hom(flux_line_above_1_marker)
    integral_flux_below_1_squish = flux_normal_squish * dS_hom(flux_line_below_1_marker)
    integral_flux_above_2_squish = flux_normal_squish * dS_hom(flux_line_above_2_marker)
    integral_flux_below_2_squish = flux_normal_squish * dS_hom(flux_line_below_2_marker)

    # Assemble the integral
    flux_total_above_1_squish = assemble_scalar(form(integral_flux_above_1_squish))
    flux_total_below_1_squish = assemble_scalar(form(integral_flux_below_1_squish))
    flux_total_above_2_squish = assemble_scalar(form(integral_flux_above_2_squish))
    flux_total_below_2_squish = assemble_scalar(form(integral_flux_below_2_squish))
    # Average flux over the line
    line_length_1_above = assemble_scalar(form(1 * dS_hom(flux_line_above_1_marker)))
    line_length_1_below = assemble_scalar(form(1 * dS_hom(flux_line_below_1_marker)))
    line_length_2_above = assemble_scalar(form(1 * dS_hom(flux_line_above_2_marker)))
    line_length_2_below = assemble_scalar(form(1 * dS_hom(flux_line_below_2_marker)))

    # Average flux over the line
    average_flux_above_1_squish = flux_total_above_1_squish/ line_length_1_above
    average_flux_below_1_squish = flux_total_below_1_squish/ line_length_1_below
    average_flux_above_2_squish = flux_total_above_2_squish/ line_length_2_above
    average_flux_below_2_squish = flux_total_below_2_squish/ line_length_2_below

    flux_full_squish_above_1_list.append(average_flux_above_1_squish)
    flux_full_squish_below_1_list.append(average_flux_below_1_squish)
    flux_full_squish_above_2_list.append(average_flux_above_2_squish)
    flux_full_squish_below_2_list.append(average_flux_below_2_squish)


    return





for m in mesh_size_list:
    homfixedmesh(m)
    fullfixedmesh(m)
    


full_flux_above_1_array = np.array(flux_full_above_1_list)
num_flux_above_1_array = np.array(flux_num_above_1_list)
an_flux_above_1_array = np.array(flux_an_above_1_list)
mesh_size_array = np.array(mesh_size_list)

np.savetxt("full_flux_above_1_array",full_flux_above_1_array)
np.savetxt("num_flux_above_1_array",num_flux_above_1_array)
np.savetxt("an_flux_above_1_array",an_flux_above_1_array)
np.savetxt("mesh_size_array", mesh_size_array)




