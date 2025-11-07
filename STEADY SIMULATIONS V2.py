### STEADY SIMULATIONS V2
#### steady simulation code with plotting for flux vs permeability and error vs permeability
import gmsh
import meshio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import time


from petsc4py import PETSc
from dolfinx import default_scalar_type, io, geometry
from dolfinx.fem import (dirichletbc, Function, functionspace, locate_dofs_topological)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile
from dolfinx.io.gmshio import model_to_mesh
from dolfinx.mesh import create_rectangle
from dolfinx.plot import vtk_mesh

from dolfinx.fem import (Constant, Function, functionspace, assemble_scalar, dirichletbc, form, locate_dofs_topological, set_bc)
from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector, set_bc)
from dolfinx.mesh import create_mesh
from ufl import ( FacetNormal, Measure, TestFunction, TrialFunction, TrialFunctions, TestFunctions,avg, dot,dx, inner, grad)

from ufl import  SpatialCoordinate
from mpi4py import MPI

import gmsh
import numpy as np
import pyvista
import gc
import atexit
from petsc4py import PETSc
import gc

comm = MPI.COMM_WORLD

n=20 # number of "impermeable chunks" of membrane not including endpoints, # of channels = n-1
R = 2.5 # radius = half height of domain
width = 5 #width of domain
delta = width/n #channel spacing
L = 0.25 #half channel length
epsilon = 1*(10**(-15)) # to use as small number

#distances from membrane to compute flux
dist_1= 0.5
dist_2 = 1.5
eps_list=[0.4,0.25,0.2,0.15,0.1,0.05,0.025] #ratio of channel width to spacing, choose values as required
chosen_eps_vals = [0.05,0.25] #eps values for plotting
levels = np.linspace(0.025,0.975,39) # contour spacing

#empty lists to store flux and error values 
permeability_list=[]

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

##create mesh to use in effective simulations - shouldn't change with membrane geometry just make once outside
# Define the desired mesh size
mesh_size = 0.008
gdim = 2
model_rank = 0
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
    # We fuse the two rectangles and keep the interface between them - mark as one domain but want interface line?
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
    gmsh.option.setNumber("Mesh.Algorithm", 2) #try algorithms 2 (automatic) or 7 (BAMG(useful when certain directions need finer refinement))
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

line_length_1_above = assemble_scalar(form(1 * dS_hom(flux_line_above_1_marker)))
line_length_1_below = assemble_scalar(form(1 * dS_hom(flux_line_below_1_marker)))
line_length_2_above = assemble_scalar(form(1 * dS_hom(flux_line_above_2_marker)))
line_length_2_below = assemble_scalar(form(1 * dS_hom(flux_line_below_2_marker)))


#create gap mesh once outside for visualisation

gdim = 2
mesh_comm = MPI.COMM_WORLD
model_rank = 0
gmsh.initialize()

###making gap mesh with physical entities on gap boundaries for proper cell alignment
if mesh_comm.rank == model_rank:
    rectangle = gmsh.model.occ.addRectangle(0, 0, 0, width, 2*R, tag=1)
    eff_mem= gmsh.model.occ.addRectangle(0,R-L,0, width, 2*L)
    
    
    
    whole_domain = gmsh.model.occ.cut([(gdim, rectangle)], [(gdim, eff_mem)])
    gmsh.model.occ.synchronize()

    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)
    # Add points for horizontal lines for gap boundaries
    p1_above = gmsh.model.occ.addPoint(0, R + L, 0)
    p2_above = gmsh.model.occ.addPoint(width, R + L, 0)
    line_above = gmsh.model.occ.addLine(p1_above, p2_above)
    gmsh.model.occ.synchronize()


    p1_below = gmsh.model.occ.addPoint(0, R - L, 0)
    p2_below = gmsh.model.occ.addPoint(width, R - L, 0)
    line_below = gmsh.model.occ.addLine(p1_below, p2_below)
    gmsh.model.occ.synchronize()

    
            
    
    # Fragment the added lines with the existing geometry
    gmsh.model.occ.fragment([(1, line_above), (1, line_below)], gmsh.model.getEntities(dim=2))

    
    # Synchronize and mesh
    gmsh.model.occ.synchronize()
    

    #set mesh size for all points
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)
    


    #tag boundaries
    top_wall=[]
    bottom_wall=[]
    side_walls=[]
    membrane_boundary=[]
    
    
    for line in gmsh.model.getEntities(dim=1):
        com = gmsh.model.occ.getCenterOfMass(line[0], line[1])
        if np.isclose(com[1],0):
            bottom_wall.append(line[1])
        elif np.isclose(com[1],2*R):
            top_wall.append(line[1])
        elif np.isclose(com[1], (R-L)/2):
            side_walls.append(line[1])
        elif np.isclose(com[1], R+L + (R-L)/2):
            side_walls.append(line[1])           
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

    gap_mesh, ct_gap, ft_gap= model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=2) #sets the mesh, cell tags and facet tags
    ft_gap.name = "Facet markers"

    gmsh.finalize()

def effective_problem(eps,n):
    delta = width/n #i.e delta=0.25

    P = 1/((L/eps)+(2*delta/np.pi)*(np.log(1/(8*eps))+1)) # defines permeability in terms of channel geometry 
    permeability_list.append(P)


    #interface condition constant
    p = Constant(mesh_hom, PETSc.ScalarType(P))

    #MIXED ELEMENT FORMULATION
    #define function space and mixed elements
    V1 = functionspace(mesh_hom, ("CG", 1))
    V2 = functionspace(mesh_hom, ("CG", 1))

    #define mixed function space
    V_hom = functionspace(mesh_hom, V1.ufl_element() * V2.ufl_element())


    #define trial and test functions
    (u_1,u_2)= TrialFunctions(V_hom)
    (w_1,w_2)= TestFunctions(V_hom)

    #MIXED FUNCTION SPACE BOUNDARY CONDITIONS AND ASSEMBLY
    #define boundary condition
    outer_below_bc = dirichletbc(default_scalar_type(1),locate_dofs_topological(V_hom.sub(0), tdim-1, ft_hom.find(bottom_wall_marker)),V_hom.sub(0)) # do i need to include mixed function space as arument to bcs if not included assumes function space is the same as that bcs are applied to??
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
    

    #solve - new method involves lines from solver.settype to solver.setoperators
    w_sol = Function(V_hom)
    solver = PETSc.KSP().create(comm)
    # Set the solver type to 'preonly' (direct solve)
    solver.setType(PETSc.KSP.Type.PREONLY)
    # Set the preconditioner type to 'lu' (direct factorization)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.LU)
    solver.setOperators(A_hom)

    A_hom.shift(1e-10) # shift solves invertibility issues
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
    # For example, setting values to zero in the upper half of the domain
    coords = mesh_hom.geometry.x
    y_coords = coords[:, 1]  # y-coordinate of the mesh vertices

    # Define a threshold to identify the upper half of the domain
    threshold = R - L - epsilon

    # Find indices where y-coordinates are greater than the threshold
    indices1 = np.where(y_coords > threshold)[0]
    indices2 = np.where(y_coords < threshold)[0]

    # Set the corresponding solution values to zero
    u1_vis_array[indices1] = 0
    u2_vis_array[indices2] = 0

    u_vis.x.array[:] = u1_vis_array + u2_vis_array


    ########################################################### ANALYTICAL SOLUTION PLOTTED #######################################################
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

    if eps in chosen_eps_vals: ###plot analytical solution for chosen values of eps

        V_gap = functionspace(gap_mesh, ("CG", 1))
        coords_gap = gap_mesh.geometry.x

         ######## INTERPOLATING ANALYTICAL
        values_gap_an = np.zeros(coords_gap.shape[0])
        coords_hom = mesh_hom.geometry.x
        u_gap_an = Function(V_gap)

         

        # Create bounding box tree for old mesh

        bb_tree_hom = geometry.bb_tree(mesh_hom,tdim)


        for i, coord in enumerate(coords_gap):
            x, y, _ = coord

            
            if y < R - L-1e-2:
                point = np.array([x, y, 0.0])

            elif np.isclose(y, R - L, atol=1e-2):
                # Force continuity with bottom region
                point = np.array([x, y - 1e-2, 0.0])  # tiny downward shift on interface as it takes value of upper half

            # Top rectangle: shift down by 2L to map back to original domain
            elif y >= R + L:
                point = np.array([x, y - 2 * L, 0.0])  # shift down by gap height

            # Inside the gap: no solution, set to zero
            else:
                values_gap_an[i] = 0.0
                continue

            # Interpolate from original mesh
            cell_candidates = geometry.compute_collisions_points(bb_tree_hom, point)
            colliding_cells = geometry.compute_colliding_cells(mesh_hom, cell_candidates, point)

            if len(colliding_cells) > 0:
                closest_cell = colliding_cells[0]
                values_gap_an[i] = u_an.eval([point], [closest_cell])[0]
            else:
                # Fallback: nearest neighbor
                distances = np.linalg.norm(coords_hom[:, :2] - point[:2], axis=1)
                nearest_idx = np.argmin(distances)
                values_gap_an[i] = u_an.x.array[nearest_idx]


        # Assign interpolated values to the new function
        u_gap_an.x.array[:] = values_gap_an

       


    

        #plot the new function on new mesh with gap
        gap_mesh.topology.create_connectivity(tdim,tdim)
        topology, cell_types, x= vtk_mesh(gap_mesh, tdim)
        grid = pyvista.UnstructuredGrid(topology, cell_types, x)
        num_local_cells = gap_mesh.topology.index_map(tdim).size_local
        grid_u = pyvista.UnstructuredGrid(*vtk_mesh(V_gap))
        num_local_cells = gap_mesh.topology.index_map(tdim).size_local
        grid_u.point_data["u_gap_an"] = u_gap_an.x.array
        grid_u.set_active_scalars("u_gap_an")

       

        p3= pyvista.Plotter(window_size=[800, 800])
        p3.reset_camera() 
        p3.add_mesh(grid_u, clim=[0, grid_u.point_data["u_gap_an"].max()], show_edges=False, show_scalar_bar=False)
        p3.add_scalar_bar(
        title="c",  
        title_font_size=30,  
        label_font_size=25,  # Font size for the numbers
        vertical = True,
        position_x=0.86,      # Position of the scalar bar
        position_y=0.165,
        width=0.15,          
        height=0.7           
    )   
        
       
        contours3 = grid_u.contour(levels)

        

        
        p3.add_mesh(contours3,color = 'black', line_width=1,scalar_bar_args=None,show_scalar_bar=False )
        p3.view_xy()
        p3.camera.zoom(1)
        p3.save_graphic("cont_int_u_gap_an_perm{}_delta_{}_eps_{}.svg".format(P,delta,eps))
        p3.close()

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

    # --- explicit cleanup (effective) ---
    try: solver.destroy()
    except: pass
    try: pc.destroy()
    except: pass
    try: A_hom.destroy()
    except: pass
    try: b_hom.destroy()
    except: pass

    # Drop references that may hold PETSc handles via forms/vectors
    del a_hom, l_hom, A_hom, b_hom, solver, pc, w_sol, u_sol, v_sol
    gc.collect()
    # Optional: small sleep to give OS a moment (not required)
    time.sleep(0.1)
    if eps in chosen_eps_vals:
        return u_an ##to use in visualising absolute error on gap mesh
    return

def full_problem(eps,n):

    #mesh generation for full simulation - will change with each epsilon
    gmsh.initialize()

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
        full_flux_line_above_1 =[]
        full_flux_line_below_1=[]
        full_flux_line_above_2 =[]
        full_flux_line_below_2=[]
        
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
                full_flux_line_below_1.append(line[1])
            elif np.isclose(com[1],R+L + dist_1):
                full_flux_line_above_1.append(line[1])
            elif np.isclose(com[1],R-L-dist_2):
                full_flux_line_below_2.append(line[1])
            elif np.isclose(com[1],R+L + dist_2):
                full_flux_line_above_2.append(line[1])
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
        gmsh.model.addPhysicalGroup(1, full_flux_line_above_1, flux_line_above_1_marker)
        gmsh.model.addPhysicalGroup(1, full_flux_line_below_1, flux_line_below_1_marker)
        gmsh.model.addPhysicalGroup(1, full_flux_line_above_2, flux_line_above_2_marker)
        gmsh.model.addPhysicalGroup(1, full_flux_line_below_2, flux_line_below_2_marker)
        
        
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

    P = 1/((L/eps)+(2*delta/np.pi)*(np.log(1/(8*eps))+1)) # defines permeability in terms of channel geometry 
    #full solution and fluxes
    V = functionspace(mesh_full, ("CG",1)) #define function space
    tdim = mesh_full.topology.dim #topological dimension of domain


    outer_below_bc = dirichletbc(default_scalar_type(1),locate_dofs_topological(V, tdim-1, ft.find(bottom_wall_marker)),V)
    outer_above_bc = dirichletbc(default_scalar_type(0), locate_dofs_topological(V, tdim-1, ft.find(top_wall_marker)),V)
    bcs = [outer_below_bc, outer_above_bc]

    u = TrialFunction(V)
    v = TestFunction(V)

    dx_full = Measure("dx", domain = mesh_full, subdomain_data= ct)

    a = dot(grad(u),grad(v))*dx_full
    L_full = Constant(mesh_full, default_scalar_type(0)) *v*dx_full

    problem = LinearProblem(a, L_full, bcs=bcs,petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()

    ####VISUALISE FULL SOLUTION FOR CHOOSEN MEMBRANE GEOMETRY
    if eps in chosen_eps_vals:
        mesh_full.topology.create_connectivity(tdim, tdim) #ensures connectivity information for tdim (cells) is availabel (i.e. how cells connect to each other)
        topology, cell_types, x = vtk_mesh(mesh_full, tdim) # extracts mesh data in a format compatible with visualisation toolkit used by pyvista topology = connectivity, x = coordinates of the vertices
        grid = pyvista.UnstructuredGrid(topology, cell_types, x) #creates unstructured grid object for visualisation with topology etc defined above
        num_local_cells = mesh_full.topology.index_map(tdim).size_local #relates to parallel computing this plus next line filter for "locally owned" cells
        grid.cell_data["Marker"] = ct.values[ct.indices < num_local_cells]
        grid.set_active_scalars("Marker")


          
        grid_uh = pyvista.UnstructuredGrid(*vtk_mesh(V))
        grid_uh.point_data["uh"] = uh.x.array.real
        grid_uh.set_active_scalars("uh")
        p2 = pyvista.Plotter(window_size=[800, 800])
        p2.add_mesh(grid_uh, clim=[0, grid_uh.point_data["uh"].max()],show_edges=False, show_scalar_bar=False )
        p2.add_scalar_bar(
        title="c",  
        title_font_size=30,  
        label_font_size=25,  
         vertical = True,
        position_x=0.86,      
        position_y=0.165,
        width=0.15,          
        height=0.7           
    )   
        contours = grid_uh.contour(levels)
        p2.add_mesh(contours, color='black', line_width=1,scalar_bar_args=None,show_scalar_bar=False )
    #set show_edges to true to see the mesh overlayed
        p2.view_xy()
        p2.camera.zoom(1)
        p2.save_graphic("full_u_flux_eps{}_delta{}_perm_{}.svg".format(eps,delta,P), raster=False) 
        p2.close()



#flux calculations in full simulation
    dS_full = Measure("dS", mesh_full, subdomain_data = ft) 
    flux_full = -grad(uh)
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


    ##interpolating full solution onto original mesh
    coords_hom = mesh_hom.geometry.x
    values_full_squish = np.zeros(coords_hom.shape[0])
    V_vis = functionspace(mesh_hom, ("CG",1))
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
    average_flux_above_1_squish = flux_total_above_1_squish/ line_length_1_above
    average_flux_below_1_squish = flux_total_below_1_squish/ line_length_1_below
    average_flux_above_2_squish = flux_total_above_2_squish/ line_length_2_above
    average_flux_below_2_squish = flux_total_below_2_squish/ line_length_2_below

    flux_full_squish_above_1_list.append(average_flux_above_1_squish)
    flux_full_squish_below_1_list.append(average_flux_below_1_squish)
    flux_full_squish_above_2_list.append(average_flux_above_2_squish)
    flux_full_squish_below_2_list.append(average_flux_below_2_squish)
   
    # --- explicit cleanup (full) ---
    try: problem.A.handle.destroy(); problem.b.handle.destroy()
    except: pass
    try: problem.ksp.destroy()
    except: pass

    # Drop objects that can own PETSc handles or references into them
    del (problem, uh, V, dS_full, dx_full, flux_full, flux_normal_full,
         integral_flux_above_1_full, integral_flux_below_1_full,
         integral_flux_above_2_full, integral_flux_below_2_full,
         mesh_full, ct, ft)


    gc.collect()
    # Optional: small sleep to give OS a moment (not required)
    time.sleep(0.1)
    if eps in chosen_eps_vals:
        return u_full_squish  ##to use in visualising absolute error on gap mesh
    return


run =0
for eps in eps_list:
    effective_problem(eps,n=20)
    full_problem(eps,n=20)
    # Output the average flux, append to relevant list
    er_num_v_an_1_above.append(abs(flux_num_above_1_list[run]-flux_an_above_1_list[run]))
    er_num_v_an_1_below.append(abs(flux_num_below_1_list[run]-flux_an_below_1_list[run]))
    er_num_v_an_2_above.append(abs(flux_num_above_2_list[run]-flux_an_above_2_list[run]))
    er_num_v_an_2_below.append(abs(flux_num_below_2_list[run]-flux_an_below_2_list[run]))
    er_num_v_full_1_above.append(abs(flux_num_above_1_list[run]-flux_full_squish_above_1_list[run]))
    er_num_v_full_1_below.append(abs(flux_num_below_1_list[run]-flux_full_squish_below_1_list[run]))
    er_num_v_full_2_above.append(abs(flux_num_above_2_list[run]-flux_full_squish_above_2_list[run]))
    er_num_v_full_2_below.append(abs(flux_num_below_2_list[run]-flux_full_squish_below_2_list[run]))
    er_an_v_full_1_above.append(abs(flux_an_above_1_list[run]-flux_full_squish_above_1_list[run]))
    er_an_v_full_1_below.append(abs(flux_an_below_1_list[run]-flux_full_squish_below_1_list[run]))
    er_an_v_full_2_above.append(abs(flux_an_above_2_list[run]-flux_full_squish_above_2_list[run]))
    er_an_v_full_2_below.append(abs(flux_an_below_2_list[run]-flux_full_squish_below_2_list[run]))
    run+=1
    ##interpolate full solution on gap mesh and visualising difference between analytical effective solution and full simulation
    if eps in chosen_eps_vals:
        P = 1/((L/eps)+(2*delta/np.pi)*(np.log(1/(8*eps))+1))
        u_full_squish = full_problem(eps,n=20)
        u_an= effective_problem(eps,n=20)
        

        

        V_gap = functionspace(gap_mesh, ("CG", 1))
        V_hom = functionspace(mesh_hom, ("CG", 1))
        coords_gap = gap_mesh.geometry.x
        u_diff_squish = Function(V_hom)
        u_diff_squish.x.array[:] = abs(u_full_squish.x.array - u_an.x.array)

        values_gap_full = np.zeros(coords_gap.shape[0])
        coords_hom = mesh_hom.geometry.x
       

  

        # Create bounding box tree for old mesh

        bb_tree_hom = geometry.bb_tree(mesh_hom,tdim)




        for i, coord in enumerate(coords_gap):
            x, y, _ = coord

            
            if y < R - L-1e-6:
                point = np.array([x, y, 0.0])

            elif np.isclose(y, R - L, atol=1e-6):
                point = np.array([x, y - 1e-6, 0.0])  # tiny downward shift

            # Top rectangle: shift down by 2L to map back to original domain
            elif y >= R + L:
                point = np.array([x, y - 2 * L, 0.0])  # shift down by gap height

            # Inside the gap: no solution, set to zero
            else:
                values_gap_full[i] = 0.0
                continue

            # Interpolate from original mesh
            cell_candidates = geometry.compute_collisions_points(bb_tree_hom, point)
            colliding_cells = geometry.compute_colliding_cells(mesh_hom, cell_candidates, point)

            if len(colliding_cells) > 0:
                closest_cell = colliding_cells[0]
                values_gap_full[i] = u_diff_squish.eval([point], [closest_cell])[0]
            else:
                # Fallback: nearest neighbor
                distances = np.linalg.norm(coords_hom[:, :2] - point[:2], axis=1)
                nearest_idx = np.argmin(distances)
                values_gap_full[i] = u_diff_squish.x.array[nearest_idx]

        

        # Assign interpolated values to the new function
        u_diff = Function(V_gap)
        u_diff.x.array[:] = values_gap_full

    

    
        #plot the new function on new mesh with gap
        gap_mesh.topology.create_connectivity(tdim,tdim)
        topology, cell_types, x= vtk_mesh(gap_mesh, tdim)
        grid = pyvista.UnstructuredGrid(topology, cell_types, x)
        num_local_cells = gap_mesh.topology.index_map(tdim).size_local
        grid_u = pyvista.UnstructuredGrid(*vtk_mesh(V_gap))
        num_local_cells = gap_mesh.topology.index_map(tdim).size_local
        grid_u.point_data["u_diff"] = u_diff.x.array
        grid_u.set_active_scalars("u_diff")
       

        p4= pyvista.Plotter(window_size=[800, 800])
        p4.reset_camera() 
        p4.add_mesh(grid_u, clim=[0, grid_u.point_data["u_diff"].max()], show_edges=False, show_scalar_bar=False)
        p4.add_scalar_bar(
        title="c",  
        title_font_size=30,  
        label_font_size=25, 
         vertical = True,
        position_x=0.86,     
        position_y=0.165,
        width=0.15,          
    )    
        
        #uncomment to add contours to error plots
        # step = grid_u.point_data["u_diff"].max()- grid_u.point_data["u_diff"].min()
        # levels = np.linspace(step/25,grid_u.point_data["u_diff"].max()-step/25 ,24)
        # contours3 = grid_u.contour(levels)
        
        # p4.add_mesh(contours3,color = 'black', line_width=1,scalar_bar_args=None,show_scalar_bar=False )
        p4.view_xy()
        p4.camera.zoom(1)
        p4.save_graphic("abs_diff_gap_perm_contours{}_delta_{}_eps_{}.svg".format(P,delta,eps))
        p4.close()

   

er_num_v_an_1_above_array = np.array(er_num_v_an_1_above)
er_num_v_an_1_below_array = np.array(er_num_v_an_1_below)
er_num_v_full_1_above_array = np.array(er_num_v_full_1_above)
er_num_v_full_1_below_array= np.array(er_num_v_full_1_below)
er_an_v_full_1_above_array=np.array(er_an_v_full_1_above)
er_an_v_full_1_below_array = np.array(er_an_v_full_1_below)
er_num_v_an_2_above_array = np.array(er_num_v_an_2_above)
er_num_v_an_2_below_array = np.array(er_num_v_an_2_below)
er_num_v_full_2_above_array=np.array(er_num_v_full_2_above)
er_num_v_full_2_below_array=np.array(er_num_v_full_2_below)
er_an_v_full_2_above_array=np.array(er_an_v_full_2_above)
er_an_v_full_2_below_array=np.array(er_an_v_full_2_below)
flux_an_above_1_list_array = np.array(flux_an_above_1_list)
flux_an_below_1_list_array = np.array(flux_an_below_1_list)
flux_full_above_1_list_array = np.array(flux_full_above_1_list)
flux_full_below_1_list_array= np.array(flux_full_below_1_list)
flux_full_squish_above_1_list_array= np.array(flux_full_squish_above_1_list)
flux_full_squish_below_1_list_array= np.array(flux_full_squish_below_1_list)
flux_num_above_1_list_array= np.array(flux_num_above_1_list)
flux_num_below_1_list_array= np.array(flux_num_below_1_list)
flux_an_above_2_list_array= np.array(flux_an_above_2_list)
flux_an_below_2_list_array= np.array(flux_an_below_2_list)
flux_full_above_2_list_array= np.array(flux_full_above_2_list)
flux_full_below_2_list_array= np.array(flux_full_below_2_list)
flux_full_squish_above_2_list_array= np.array(flux_full_squish_above_2_list)
flux_full_squish_below_2_list_array= np.array(flux_full_squish_below_2_list)
flux_num_above_2_list_array= np.array(flux_num_above_2_list)
flux_num_below_2_list_array= np.array(flux_num_below_2_list)
permeability_array = np.array(permeability_list)

np.savetxt("er_num_v_an_1_above_array",er_num_v_an_1_above_array)
np.savetxt("er_num_v_an_1_below_array", er_num_v_an_1_below_array)
np.savetxt("er_num_v_full_1_above_array",er_num_v_full_1_above_array)
np.savetxt("er_num_v_full_1_below_array",er_num_v_full_1_below_array)
np.savetxt("er_an_v_full_1_above_array",er_an_v_full_1_above_array)
np.savetxt("er_an_v_full_1_below_array",er_an_v_full_1_below_array)
np.savetxt("er_num_v_an_2_above_array",er_num_v_an_2_above_array)
np.savetxt("er_num_v_an_2_below_array",er_num_v_an_2_below_array)
np.savetxt("er_num_v_full_2_above_array",er_num_v_full_2_above_array)
np.savetxt("er_num_v_full_2_below_array",er_num_v_full_2_below_array)
np.savetxt("er_an_v_full_2_above_array",er_an_v_full_2_above_array)
np.savetxt("er_an_v_full_2_below_array",er_an_v_full_2_below_array)
np.savetxt("flux_an_above_1_list_array",flux_an_above_1_list_array)
np.savetxt("flux_an_below_1_list_array",flux_an_below_1_list_array)
np.savetxt("flux_full_above_1_list_array",flux_full_above_1_list_array)
np.savetxt("flux_full_below_1_list_array",flux_full_below_1_list_array)
np.savetxt("flux_full_squish_above_1_list_array",flux_full_squish_above_1_list_array)
np.savetxt("flux_full_squish_below_1_list_array",flux_full_squish_below_1_list_array)
np.savetxt("flux_num_above_1_list_array",flux_num_above_1_list_array)
np.savetxt("flux_num_below_1_list_array",flux_num_below_1_list_array)
np.savetxt("flux_an_above_2_list_array",flux_an_above_2_list_array)
np.savetxt("flux_an_below_2_list_array",flux_an_below_2_list_array)
np.savetxt("flux_full_above_2_list_array",flux_full_above_2_list_array)
np.savetxt("flux_full_below_2_list_array2",flux_full_below_2_list_array)
np.savetxt("flux_full_squish_above_2_list_array",flux_full_squish_above_2_list_array)
np.savetxt("flux_full_squish_below_2_list_array",flux_full_squish_below_2_list_array)
np.savetxt("flux_num_above_2_list_array",flux_num_above_2_list_array)
np.savetxt("flux_num_below_2_list_array",flux_num_below_2_list_array)
np.savetxt("permeability_array", permeability_array)

