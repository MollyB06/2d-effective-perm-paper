#### running full and effective in different timestepping loop0s to avoid RAM errors
##### TIME DEPENDENT SIMULATIONS
from math import exp, log, pi,sqrt
import gmsh
import meshio
import numpy as np
import matplotlib.pyplot as plt
from petsc4py import PETSc
from dolfinx import default_scalar_type, io, geometry, plot
from dolfinx.fem import (dirichletbc, Function, functionspace, locate_dofs_topological)
from dolfinx.io import XDMFFile
from dolfinx.io.gmshio import model_to_mesh
from dolfinx.mesh import create_rectangle
from dolfinx.fem import (Constant, Function, functionspace,assemble_scalar, dirichletbc, form, locate_dofs_topological, set_bc)
from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector,create_vector, set_bc)
from dolfinx.mesh import create_mesh
from ufl import (Measure, TestFunction, TrialFunction, TrialFunctions, TestFunctions, avg, inner, grad)
from ufl import  SpatialCoordinate
from mpi4py import MPI

import gmsh
import numpy as np
import pyvista
import ufl


comm = MPI.COMM_WORLD


#PARAMETERS FOR BOTH SIMULATIONS


 # size of time step

epsilon = 1*(10**(-15)) ##TO USE AS SMALL NUMBER

# Define the desired mesh size
mesh_size = 0.008  #note if want to generate gifs for time dependent sims may need to reduce resolution to deal with RAM issues

#distance of flux lines from membrane
dist = 1

#domain parameters
R = 2.5 # radius = half height of domain
width = 5 #width of domain
L = 0.25

def lambda_m(m):
    return (m*pi)/L


trunc = 30 # number of terms we take in sum before truncating






def time_sim_effective(eps,n_chunk=20,num_steps=48, final_time=12):
    t = 0 # intial time
    T = final_time#final time

    delt = (T-t)/num_steps
    delta = width/n_chunk 
    ## for naming purposes
    if eps==0.25:
        run=1
    elif eps==0.05:
        run=2

    P = 1/((L/eps)+(2*delta/np.pi)*(np.log(1/(8*eps))+1)) # defines permeability in terms of channel geometry 

    
    ########################## MESH FOR EFFECTIVE SIMULATION
    gdim = 2
    model_rank = 0

    gmsh.initialize()


    proc = MPI.COMM_WORLD.rank
    mesh_comm = MPI.COMM_WORLD


    #define markers for domains
    main_marker = 2
    interface_marker = 3
    top_wall_marker=4
    bottom_wall_marker=5
    bottom_sub_marker= 6
    top_sub_marker = 7
    flux_line_above_marker=10
    flux_line_below_marker=11



    if proc == 0:
        # We create one rectangle for each subdomain
        rec1 = gmsh.model.occ.addRectangle(0, 0, 0, width, R-L, tag=1)
        rec2 = gmsh.model.occ.addRectangle(0, R-L, 0, width, R-L, tag=2)
        # We fuse the two rectangles and keep the interface between them - mark as one domain but want interface line?
        gmsh.model.occ.fragment([(2, rec1)], [(2, rec2)])
        gmsh.model.occ.synchronize()
        

        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)
        # Add points for horizontal lines to calculate flux over
        p1_above = gmsh.model.occ.addPoint(0, R - L + dist, 0)
        p2_above = gmsh.model.occ.addPoint(width, R - L + dist, 0)
        line_above = gmsh.model.occ.addLine(p1_above, p2_above)
        gmsh.model.occ.synchronize()


        p1_below = gmsh.model.occ.addPoint(0, R - L - dist, 0)
        p2_below = gmsh.model.occ.addPoint(width, R - L - dist, 0)
        line_below = gmsh.model.occ.addLine(p1_below, p2_below)
        
        
        # Fragment the added lines with the existing geometry
        gmsh.model.occ.fragment([(1, line_above), (1, line_below)], gmsh.model.getEntities(dim=2))
        
        gmsh.model.occ.synchronize()

        # Synchronize and mesh
        gmsh.model.occ.synchronize()
        

        #set mesh size for all points
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)

        #mark different boundaries
        top_wall=[]
        bottom_wall=[]
        side_walls=[]
        interface = []
        flux_line_above_hom =[]
        flux_line_below_hom =[]
        for line in gmsh.model.getEntities(dim=1):
            com = gmsh.model.occ.getCenterOfMass(line[0], line[1])
            if np.isclose(com[1],0):
                bottom_wall.append(line[1])
            elif np.isclose(com[1],2*R-2*L):
                top_wall.append(line[1])
            elif np.isclose(com[1],R-L):
                interface.append(line[1])
            elif np.isclose(com[1], (R-L-dist)/2):
                side_walls.append(line[1])
            elif np.isclose(com[1], (3*(R-L)+dist)/2):
                side_walls.append(line[1])
            elif np.isclose(com[1],R-L-dist/2):
                side_walls.append(line[1])
            elif np.isclose(com[1], R-L+dist/2):
                side_walls.append(line[1])
            elif np.isclose(com[1],R-L-dist):
                flux_line_below_hom.append(line[1])
            elif np.isclose(com[1],R-L + dist):
                flux_line_above_hom.append(line[1])
                

        #mark subdomains
        
        
        bottom_half= []
        top_half =[]
        for surface in gmsh.model.getEntities(dim=2):
            com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
            if np.allclose(com, [width/2, (R-L-dist)/2, 0]):
                bottom_half.append(surface[1])
            elif np.allclose(com, [width/2, R-L-dist/2,0]):
                bottom_half.append(surface[1])
            elif np.allclose(com, [width/2, (3*(R-L)+dist)/2, 0]):
                top_half.append(surface[1]) 
            elif np.allclose(com, [width/2, R-L +dist/2,0]):
                top_half.append(surface[1])
        
        # Synchronize and mesh
        gmsh.model.occ.synchronize()   
            

        gmsh.model.addPhysicalGroup(2, top_half, top_sub_marker)
        gmsh.model.addPhysicalGroup(2, bottom_half, bottom_sub_marker)
        gmsh.model.addPhysicalGroup(1, top_wall, top_wall_marker)
        gmsh.model.addPhysicalGroup(1, bottom_wall, bottom_wall_marker)
        gmsh.model.addPhysicalGroup(1, interface, interface_marker)
        gmsh.model.addPhysicalGroup(1, flux_line_above_hom, flux_line_above_marker)
        gmsh.model.addPhysicalGroup(1, flux_line_below_hom, flux_line_below_marker)
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



    MPI.COMM_WORLD.barrier()


    tdim = mesh_hom.topology.dim #topological dimension of domain

    #define integration measures
    dx_hom = Measure("dx", domain = mesh_hom, subdomain_data= ct_hom)
    dS_hom = Measure("dS", mesh_hom, subdomain_data = ft_hom) 

    n=0 # time step

    
    #MIXED ELEMENT FORMULATION
    #define function space and mixed elements
    V1 = functionspace(mesh_hom, ("CG", 1))
    V2 = functionspace(mesh_hom, ("CG", 1))

    #define mixed function space
    V_hom_t = functionspace(mesh_hom, V1.ufl_element() * V2.ufl_element()) #mixed function space
    V_nm_t = functionspace(mesh_hom, ("CG",1)) #non mixed function space




    x = SpatialCoordinate(mesh_hom)
    #MIXED FUNCTION SPACE BOUNDARY CONDITIONS AND ASSEMBLY
    #define boundary condition
    outer_below_bc = dirichletbc(default_scalar_type(1),locate_dofs_topological(V_hom_t.sub(0), tdim-1, ft_hom.find(bottom_wall_marker)),V_hom_t.sub(0)) 
    outer_above_bc = dirichletbc(default_scalar_type(0), locate_dofs_topological(V_hom_t.sub(1), tdim-1, ft_hom.find(top_wall_marker)),V_hom_t.sub(1))

    bc = [outer_below_bc, outer_above_bc]


    # Manipulate values so that function is zero where it shouldn't be defined
    coords = mesh_hom.geometry.x


    #define intial condition
    def initial_condition_1(x):
        return np.exp(-1000*x[1])


    def initial_condition_2(x):
        return 0*x[1]

    #interpolate initial condition onto function space
    #define intial conditions for each half of the domain
    u_0 = Function(V_hom_t)
    u_0.sub(0).interpolate(initial_condition_1)
    u_0.sub(1).interpolate(initial_condition_2)

    chi = ((2*delta*eps)/pi)*(log(1/(8*eps))+1)


    #lists to store function values and flux values at each time step
    u_1_0,u_2_0 = u_0.split()
    u_1_init = Function(V_nm_t)
    u_2_init = Function(V_nm_t)
    u_1_init.interpolate(u_1_0)
    u_2_init.interpolate(u_2_0)


    ##used to store previous time u1 and u2 values to use in interface condition integral
    u_1_list=[u_1_init] 
    u_2_list=[u_2_init]
    du_1 = Function(V_nm_t)
    du_2 = Function(V_nm_t)
    du_1.x.array[:] = 0.0
    du_2.x.array[:] = 0.0
    du_1_list=[du_1]
    du_2_list=[du_2]
    times_list=[num*delt for num in range(num_steps+1)]
    expvalues =[exp(time) for time in times_list]

    #interpolate intial condition to current and working u
    u_n = Function(V_hom_t)
    u_n.interpolate(u_0)
    uh = Function(V_hom_t)
    uh.interpolate(u_0)
    u_1_n, u_2_n = u_n.split() #split current solution into its values in each part of the domain


    #define trial and test functions
    (u1,u2)= TrialFunctions(V_hom_t)
    (v1,v2)= TestFunctions(V_hom_t)


    #define variational form

    H_list=[]
    F_list=[]
    Hm2 = Function(V_nm_t)
    Fm2 = Function(V_nm_t)
    Hm2.x.array[:] = 0.0
    Fm2.x.array[:]= 0.0
    H_list.append(Hm2)
    F_list.append(Fm2)

    Hm1 = Function(V_nm_t)
    Fm1 = Function(V_nm_t)
    Hm1.x.array[:] = 0.0
    Fm1.x.array[:]= 0.0
    H_list.append(Hm1)
    F_list.append(Fm1)

    #jump condition to be applied across interface
    #interface condition constant

    p = Constant(mesh_hom, PETSc.ScalarType(P))
    jmp_u = avg(u2) - avg(u1)
    jmp_v = avg(v2)- avg(v1) 
    plus_v = avg(v1) + avg(v2)
    a_hom= (u1*v1 + delt*inner(grad(u1),grad(v1)))*dx_hom(bottom_sub_marker) + (u2*v2 + delt*inner(grad(u2),grad(v2)))*dx_hom(top_sub_marker) + delt*p*jmp_v*jmp_u*dS_hom(interface_marker)

    #interfaceconditions
    #make list for sums up to truncation value and then sum lists
    def even_sum(t,n):
        even_sum_list =[4/(lambda_m(2*m))**2*(F_list[n+2].x.array - F_list[n+1].x.array-sum(exp((lambda_m(2*m))**2*(times_list[i]-t)/4)*(F_list[i+2].x.array-2*F_list[i+1].x.array+F_list[i].x.array) for i in range(1,n)) + 0.5*F_list[n+2].x.array - F_list[n+1].x.array +0.5*F_list[n].x.array )for m in range(1,trunc)]
        return sum(even_sum_list)

    def odd_sum(t,n):
        odd_sum_list = [4/(lambda_m(2*m+1))**2*(H_list[n+2].x.array - H_list[n+1].x.array-sum(exp((lambda_m(2*m-1))**2*(times_list[i]-t)/4)*(H_list[i+2].x.array-2*H_list[i+1].x.array+H_list[i].x.array) for i in range(1,n)) + 0.5*H_list[n+2].x.array - H_list[n+1].x.array +0.5*H_list[n].x.array )for m in range(trunc)]
        return sum(odd_sum_list)


    even = Function(V_nm_t)
    odd = Function(V_nm_t)


    bilinear_form = form(a_hom)
    A_hom = assemble_matrix(bilinear_form, bc) #assembles matrix associated with the bilinear form
    A_hom.assemble()



    l_hom= u_1_n*v1*dx_hom(bottom_sub_marker) + u_2_n*v2*dx_hom(top_sub_marker) +(Constant(mesh_hom, PETSc.ScalarType((2*eps)/(L + chi)))*jmp_v*avg(even) - Constant(mesh_hom, PETSc.ScalarType((2*eps)))*plus_v*avg(odd))*dS_hom(interface_marker)
    linear_form = form(l_hom)
    b_hom = create_vector(linear_form) # initilaises right hand side based on linear form L (current solution u_n is initial condition)


    # Create solver
    solver_hom = PETSc.KSP().create(mesh_hom.comm)


    # Set solver type 
    # solver_hom.setType(PETSc.KSP.Type.CG)
    solver_hom.setType(PETSc.KSP.Type.GMRES)


    # Set preconditioner to LU
    pc = solver_hom.getPC()
    pc.setType(PETSc.PC.Type.LU)

    # Set operator matrix
    A_hom.shift(1e-10)  # Improves invertibility
    solver_hom.setOperators(A_hom)

    # Set solver options
    solver_hom.setTolerances(rtol=1e-6, atol=1e-10, max_it=1000)


    

    #define new functionspace for visualisation
    #interpolate to solve
    V_vis_t = functionspace(mesh_hom, ("CG",1))
    u_1_vis = Function(V_vis_t)
    u_2_vis = Function(V_vis_t)




    #split mixed function for plotting and set relevant indices to 0
    u1_sol, u2_sol= uh.split()
    u_1_vis.interpolate(u1_sol)
    u_2_vis.interpolate(u2_sol)
    u_vis = Function(V_vis_t)
    du_1_sol = Function(V_vis_t)
    du_2_sol = Function(V_vis_t)


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


    u_vis.vector[:] = u1_vis_array + u2_vis_array

    #for gif generation
    # grid_hom = pyvista.UnstructuredGrid(*plot.vtk_mesh(V_vis_t))

    # plotter_hom = pyvista.Plotter()
    # pyvista.start_xvfb()
    # plotter_hom.open_gif("u_hom_time_flat_iterativev2{}.gif".format(P), fps=10) #records a gif of solution evolving
    # grid_hom.point_data["u_vis"] = u_vis.x.array #makes current solution input for the plotter
    # #warped = grid.warp_by_scalar("uh", factor=0)

    # viridis = mpl.colormaps.get_cmap("viridis")
    # sargs = dict(title = 'c', title_font_size=30, label_font_size=25, color="black",
    #             position_x=0.15, position_y=0.01, width=0.7, height=0.2)




    # renderer = plotter_hom.add_mesh(grid_hom, show_edges=False, lighting=False,
    #                             cmap=viridis, scalar_bar_args=sargs,
    #                             clim=[0, max(u_vis.x.array)])

    # text_actor_hom = plotter_hom.add_text("0", position = "upper_left", font_size=30, color="black")

    # plotter_hom.camera.zoom(1) 
    # plotter_hom.view_xy()


    # #define new mesh and functionspace to have gap
    # gap_mesh = create_rectangle(comm, [[0,0],[width, 2*R]],[200,200])

    # #create functionspace on new mesh
    # V_gap = functionspace(gap_mesh, ("CG",1))

    # #access coordinates of new mesh
    # coords_gap = gap_mesh.geometry.x

    # #interpolate solutions onto mesh
    # #access coordinates of old mesh
    # coords_hom = mesh_hom.geometry.x

    # #create emtpy array to store function values in
    # values_gap_hom = np.zeros(coords_gap.shape[0])



    # #create function in new functionspace to interpolate solution onto
    # u_gap_hom = Function(V_gap)


    # #create bounding box tree for old mesh to allow for quick searching
    # bb_tree_hom = geometry.bb_tree(mesh_hom, tdim)

    # #loop over each point to interpolate values
    # for i, coord in enumerate(coords_gap):
    #     x,y,_ = coord
    #     if y < R-L: #for y values below membrane can use original values as is
    #         point = np.array([x,y,0])
    #         cell_candidates = geometry.compute_collisions_points(bb_tree_hom,point)
    #         colliding_cells = geometry.compute_colliding_cells(mesh_hom, cell_candidates, point)
    #         if len(colliding_cells)>0:
    #             closest_cell = colliding_cells[0]
    #             values_gap_hom[i] = u_vis.eval([point],[closest_cell])[0]
    #         else:
    #             values_gap_hom[i]=0
    #     elif y> R + L:
    #         shifted_coord = (x, y-2*L,0)
    #         cell_candidates = geometry.compute_collisions_points(bb_tree_hom,shifted_coord)
    #         colliding_cells = geometry.compute_colliding_cells(mesh_hom,cell_candidates,shifted_coord)
    #         if len(colliding_cells)>0:
    #             closest_cell= colliding_cells[0]
    #             values_gap_hom[i] = u_vis.eval([shifted_coord],[closest_cell])[0]
    #         else:
    #             values_gap_hom[i]=0
    #     else:
    #         values_gap_hom[i]=0


    # u_gap_hom.x.array[:] =values_gap_hom





    # grid_gap = pyvista.UnstructuredGrid(*plot.vtk_mesh(V_gap))

    # plotter_gap= pyvista.Plotter()
    # pyvista.start_xvfb()
    # plotter_gap.open_gif("u_hom_time_gap_iterativev2{}.gif".format(P), fps=10) #records a gif of solution evolving


    # grid_gap.point_data["u_gap_hom"] = u_gap_hom.x.array #makes current solution input for the plotter
    # #warped = grid.warp_by_scalar("uh", factor=0)

    # viridis = mpl.colormaps.get_cmap("viridis")
    # sargs = dict(title = 'c', title_font_size=30, label_font_size=25, color="black",
    #             position_x=0.15, position_y=0.01, width=0.7, height=0.2)

    # # renderer = plotter.add_mesh(warped, show_edges=False, lighting=False,
    # #                             cmap=viridis, scalar_bar_args=sargs,
    # #                             clim=[0, max(uh.x.array)])

    # renderer_gap = plotter_gap.add_mesh(grid_gap, show_edges=False, lighting=False,
    #                             cmap=viridis, scalar_bar_args=sargs,
    #                             clim=[0, max(u_vis.x.array)])


    # text_actor_gap = plotter_gap.add_text("0", position = "upper_left", font_size=30, color="black")


    # plotter_gap.camera.zoom(1) 
    # plotter_gap.view_xy()

    ##################################TIME STEPPING LOOP  EFFECTIVE INSIDE
    

    ##empty lists to store errors and times
    times_list_round = []
    hom_flux_above_list=[]
    hom_flux_below_list=[]


    for i in range(num_steps):
        print(n,t)

        Hn = Function(V_nm_t)
        Hn.x.array[:]= ((u_2_list[n].x.array + u_1_list[n].x.array)/L - chi/(2*eps)*(du_2_list[n].x.array - du_1_list[n].x.array))
        H_list.append(Hn)
        Fn = Function(V_nm_t)
        Fn.x.array[:] = (u_2_list[n].x.array - u_1_list[n].x.array)/L - chi/(2*eps)*(du_2_list[n].x.array + du_1_list[n].x.array)
        F_list.append(Fn)

        even.x.array[:] = even_sum(t,n)[:]
        odd.x.array[:] =  odd_sum(t,n)[:]


        ###NUMERICAL SOLUTION########
        # Update the right hand side reusing the initial vector
        with b_hom.localForm() as loc_b:
            loc_b.set(0) #reinitialises to 0 before updating
        assemble_vector(b_hom, linear_form)
        # Apply Dirichlet boundary condition to the vector
        apply_lifting(b_hom, [bilinear_form], [bc])
        b_hom.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b_hom, bc)

        # Solve linear problem
        solver_hom.solve(b_hom, uh.vector) #solves A.uh = b storing solution in uh
        uh.x.scatter_forward() 
        
        # Update solution at previous time step (u_n)
        u_n.x.array[:] = uh.x.array #allows for use in linear form
        



        ahom,ghom = Function(V_hom_t)
        ahom,ghom = uh.split()
        u1_sol.x.array[:], u2_sol.x.array[:] = ahom.x.array, ghom.x.array
        


        #interpolate new values for visualisation
        u_1_vis.interpolate(u1_sol)
        u_2_vis.interpolate(u2_sol)


        u1_vis_array = u_1_vis.x.array
        u2_vis_array =u_2_vis.x.array


        # Set the corresponding solution values to zero
        u1_vis_array[indices1] = 0
        u2_vis_array[indices2] = 0
        u_1_vis.x.array[indices1] = 0
        u_2_vis.x.array[indices2] = 0
        
        du_1_sol.x.array[:] = eps/(L+chi)*(u1_vis_array - u2_vis_array)-2*eps*(1/delt)*odd_sum(t,n)[:] +(2*eps)/(L+chi)*(1/delt)*even_sum(t,n)[:]
        du_2_sol.x.array[:] = eps/(L+chi)*(u1_vis_array - u2_vis_array)+2*eps*(1/delt)*odd_sum(t,n)[:] +(2*eps)/(L+chi)*(1/delt)*even_sum(t,n)[:]

        

        u_1_list.append(u_1_vis)
        u_2_list.append(u_2_vis)
        du_1_list.append(du_1_sol)
        du_2_list.append(du_2_sol)

    
        u_vis.x.array[:] = u1_vis_array + u2_vis_array 


        ##updating gif
        # plotter_hom.update_scalars(u_vis.x.array, render=False) #updates visualisation
        # plotter_hom.remove_actor(text_actor_hom)


        # # Add the updated text
        # text_actor_hom = plotter_hom.add_text(f"{t:.3f}", position = "upper_left", font_size=30, color="black")

        # plotter_hom.write_frame()

        


        # ###to visualise on gap!

        # #loop over each point to interpolate values
        # for i, coord in enumerate(coords_gap):
        #     x,y,_ = coord
        #     if y < R-L: #for y values below membrane can use original values as is
        #         point = np.array([x,y,0])
        #         cell_candidates = geometry.compute_collisions_points(bb_tree_hom,point)
        #         colliding_cells = geometry.compute_colliding_cells(mesh_hom, cell_candidates, point)
        #         if len(colliding_cells)>0:
        #             closest_cell = colliding_cells[0]
        #             values_gap_hom[i] = u_vis.eval([point],[closest_cell])[0]

        #         else:
        #             values_gap_hom[i]=0
        #     elif y> R + L:
        #         shifted_coord = (x, y-2*L,0)
        #         cell_candidates = geometry.compute_collisions_points(bb_tree_hom,shifted_coord)
        #         colliding_cells = geometry.compute_colliding_cells(mesh_hom,cell_candidates,shifted_coord)
        #         if len(colliding_cells)>0:
        #             closest_cell= colliding_cells[0]
        #             values_gap_hom[i] = u_vis.eval([shifted_coord],[closest_cell])[0]
        #         else:
        #             values_gap_hom[i]=0

        #     else:
        #         values_gap_hom[i]=0


        # u_gap_hom.x.array[:] =values_gap_hom


        # # xdmf.write_function(u_gap_hom, t)
        # plotter_gap.update_scalars(u_gap_hom.x.array, render=False) #updates visualisation
        # # Update the title
        # # Remove the old text
        # plotter_gap.remove_actor(text_actor_gap)

        # # Add the updated text
        # text_actor_gap = plotter_gap.add_text(f"{t:.3f}", position = "upper_left", font_size=30, color="black")
        # plotter_gap.write_frame()

        
        
        ####### FLUX FOR ERROR CALCULATION
         #NUMERICAL SOLUTION OF EFFECTIVE
        # Assuming u_vis is a finite element function
        flux = -grad(u_vis)

        

        # # Normal vector (facet normal)
        # norm = FacetNormal(mesh_hom)  # Domain-specific facet normal

        # Restrict flux to one side of the facet
        flux_normal_hom = avg(flux[1])


        # Define the integral using the restricted flux
        integral_flux_above_hom = flux_normal_hom * dS_hom(flux_line_above_marker)
        integral_flux_below_hom = flux_normal_hom * dS_hom(flux_line_below_marker)


        # Assemble the integral
        flux_total_above_hom = assemble_scalar(form(integral_flux_above_hom))
        flux_total_below_hom = assemble_scalar(form(integral_flux_below_hom))


        # Average flux over the line
        line_length_above = assemble_scalar(form(1 * dS_hom(flux_line_above_marker)))
        line_length_below = assemble_scalar(form(1 * dS_hom(flux_line_below_marker)))
        average_flux_above_hom = flux_total_above_hom / line_length_above
        average_flux_below_hom = flux_total_below_hom / line_length_below
        hom_flux_above_list.append(average_flux_above_hom)
        hom_flux_below_list.append(average_flux_below_hom)


       
    
        times_list_round.append(round(t,3))


        #update time
        t += delt
        n += 1
        print(t, times_list[n])


    #closing plotters once time loop is finished
    # plotter_hom.close()
    # plotter_gap.close()




    #####reformatting - save lists as np arrays to save as txt files more easily
    times_list_roundlongarray = np.array(times_list_round)
    hom_flux_above_array_long= np.array(hom_flux_above_list)
    hom_flux_below_array_long =np.array(hom_flux_below_list)
    





    ###save as txt files to call into plotting code
    np.savetxt('times_list_long_{}'.format(run),times_list_roundlongarray)
    np.savetxt('hom_flux_above_array_long_{}'.format(run), hom_flux_above_array_long)
    np.savetxt('hom_flux_below_array_long_{}'.format(run), hom_flux_below_array_long)



    return

def time_sim_full(eps,n_chunk=20,num_steps=48, final_time=12):
    t = 0 # intial time
    T = final_time#final time

    delt = (T-t)/num_steps
    delta = width/n_chunk 
    ## for naming purposes
    if eps==0.25:
        run=1
    elif eps==0.05:
        run=2

    P = 1/((L/eps)+(2*delta/np.pi)*(np.log(1/(8*eps))+1)) # defines permeability in terms of channel geometry 
    times_list=[num*delt for num in range(num_steps+1)]
    
    ########################## MESH FOR EFFECTIVE SIMULATION -used in flux calcs
    gdim = 2
    model_rank = 0

    gmsh.initialize()


    proc = MPI.COMM_WORLD.rank
    mesh_comm = MPI.COMM_WORLD


    #define markers for domains
    main_marker = 2
    interface_marker = 3
    top_wall_marker=4
    bottom_wall_marker=5
    bottom_sub_marker= 6
    top_sub_marker = 7
    flux_line_above_marker=10
    flux_line_below_marker=11



    if proc == 0:
        # We create one rectangle for each subdomain
        rec1 = gmsh.model.occ.addRectangle(0, 0, 0, width, R-L, tag=1)
        rec2 = gmsh.model.occ.addRectangle(0, R-L, 0, width, R-L, tag=2)
        # We fuse the two rectangles and keep the interface between them - mark as one domain but want interface line?
        gmsh.model.occ.fragment([(2, rec1)], [(2, rec2)])
        gmsh.model.occ.synchronize()
        

        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)
        # Add points for horizontal lines to calculate flux over
        p1_above = gmsh.model.occ.addPoint(0, R - L + dist, 0)
        p2_above = gmsh.model.occ.addPoint(width, R - L + dist, 0)
        line_above = gmsh.model.occ.addLine(p1_above, p2_above)
        gmsh.model.occ.synchronize()


        p1_below = gmsh.model.occ.addPoint(0, R - L - dist, 0)
        p2_below = gmsh.model.occ.addPoint(width, R - L - dist, 0)
        line_below = gmsh.model.occ.addLine(p1_below, p2_below)
        
        
        # Fragment the added lines with the existing geometry
        gmsh.model.occ.fragment([(1, line_above), (1, line_below)], gmsh.model.getEntities(dim=2))
        
        gmsh.model.occ.synchronize()

        # Synchronize and mesh
        gmsh.model.occ.synchronize()
        

        #set mesh size for all points
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)

        #mark different boundaries
        top_wall=[]
        bottom_wall=[]
        side_walls=[]
        interface = []
        flux_line_above_hom =[]
        flux_line_below_hom =[]
        for line in gmsh.model.getEntities(dim=1):
            com = gmsh.model.occ.getCenterOfMass(line[0], line[1])
            if np.isclose(com[1],0):
                bottom_wall.append(line[1])
            elif np.isclose(com[1],2*R-2*L):
                top_wall.append(line[1])
            elif np.isclose(com[1],R-L):
                interface.append(line[1])
            elif np.isclose(com[1], (R-L-dist)/2):
                side_walls.append(line[1])
            elif np.isclose(com[1], (3*(R-L)+dist)/2):
                side_walls.append(line[1])
            elif np.isclose(com[1],R-L-dist/2):
                side_walls.append(line[1])
            elif np.isclose(com[1], R-L+dist/2):
                side_walls.append(line[1])
            elif np.isclose(com[1],R-L-dist):
                flux_line_below_hom.append(line[1])
            elif np.isclose(com[1],R-L + dist):
                flux_line_above_hom.append(line[1])
                

        #mark subdomains
        
        
        bottom_half= []
        top_half =[]
        for surface in gmsh.model.getEntities(dim=2):
            com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
            if np.allclose(com, [width/2, (R-L-dist)/2, 0]):
                bottom_half.append(surface[1])
            elif np.allclose(com, [width/2, R-L-dist/2,0]):
                bottom_half.append(surface[1])
            elif np.allclose(com, [width/2, (3*(R-L)+dist)/2, 0]):
                top_half.append(surface[1]) 
            elif np.allclose(com, [width/2, R-L +dist/2,0]):
                top_half.append(surface[1])
        
        # Synchronize and mesh
        gmsh.model.occ.synchronize()   
            

        gmsh.model.addPhysicalGroup(2, top_half, top_sub_marker)
        gmsh.model.addPhysicalGroup(2, bottom_half, bottom_sub_marker)
        gmsh.model.addPhysicalGroup(1, top_wall, top_wall_marker)
        gmsh.model.addPhysicalGroup(1, bottom_wall, bottom_wall_marker)
        gmsh.model.addPhysicalGroup(1, interface, interface_marker)
        gmsh.model.addPhysicalGroup(1, flux_line_above_hom, flux_line_above_marker)
        gmsh.model.addPhysicalGroup(1, flux_line_below_hom, flux_line_below_marker)
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



    MPI.COMM_WORLD.barrier()


    tdim = mesh_hom.topology.dim #topological dimension of domain

    #define integration measures
    dx_hom = Measure("dx", domain = mesh_hom, subdomain_data= ct_hom)
    dS_hom = Measure("dS", mesh_hom, subdomain_data = ft_hom) 

    n=0 # time step

    
    


    x = SpatialCoordinate(mesh_hom)
    #MIXED FUNCTION SPACE BOUNDARY CONDITIONS AND ASSEMBLY
    

    ######################      MESH AND SOLVER FOR FULL SIMULATION   #############


    gmsh.initialize()
    membrane_corners_x = [delta*eps + delta/2 + i*delta for i in range(0,n_chunk-1)]
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
        p1_above = gmsh.model.occ.addPoint(0, R + L + dist, 0)
        p2_above = gmsh.model.occ.addPoint(width, R + L + dist, 0)
        line_above = gmsh.model.occ.addLine(p1_above, p2_above)
        gmsh.model.occ.synchronize()


        p1_below = gmsh.model.occ.addPoint(0, R - L - dist, 0)
        p2_below = gmsh.model.occ.addPoint(width, R - L - dist, 0)
        line_below = gmsh.model.occ.addLine(p1_below, p2_below)
        
        
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
        flux_line_above =[]
        flux_line_below =[]
        for line in gmsh.model.getEntities(dim=1):
            com = gmsh.model.occ.getCenterOfMass(line[0], line[1])
            if np.isclose(com[1],0):
                bottom_wall.append(line[1])
            elif np.isclose(com[1],2*R):
                top_wall.append(line[1])
            elif np.isclose(com[1], (R-L-dist)/2):
                side_walls.append(line[1])
            elif np.isclose(com[1], (3*R+L+dist)/2):
                side_walls.append(line[1])
            elif np.isclose(com[1],R-L-dist/2):
                side_walls.append(line[1])
            elif np.isclose(com[1], R+L+dist/2):
                side_walls.append(line[1])
            elif np.isclose(com[1],R-L-dist):
                flux_line_below.append(line[1])
            elif np.isclose(com[1],R+L + dist):
                flux_line_above.append(line[1])
            else:
                membrane_boundary.append(line[1])

        #add tag for domain
        domain_volume=[]
        for surface in gmsh.model.getEntities(dim=2):
            com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
            domain_volume.append(surface[1])



        
        #set tags
        top_wall_marker=4
        bottom_wall_marker=5
        side_walls_marker=13
        membrane_walls_marker=14
        flux_line_above_marker=10
        flux_line_below_marker=11

        #add membranes as physical groups in the mesh
        gmsh.model.addPhysicalGroup(2, domain_volume,0)
        gmsh.model.addPhysicalGroup(1, top_wall, top_wall_marker)
        gmsh.model.addPhysicalGroup(1, bottom_wall, bottom_wall_marker)
        gmsh.model.addPhysicalGroup(1, membrane_boundary, membrane_walls_marker)
        gmsh.model.addPhysicalGroup(1, flux_line_above,flux_line_above_marker)
        gmsh.model.addPhysicalGroup(1,flux_line_below,flux_line_below_marker)
        
        
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

    mesh_full, ct_full, ft_full= model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=2) #sets the mesh, cell tags and facet tags
    ft_full.name = "Facet markers"

    gmsh.finalize()

    with XDMFFile(MPI.COMM_WORLD, "mt.xdmf", "w") as xdmf_full:
        xdmf_full.write_mesh(mesh_full)
        xdmf_full.write_meshtags(ct_full, mesh_full.geometry)



    tdim = mesh_full.topology.dim #topological dimension of domain

    #define function space
    V_full_t = functionspace(mesh_full, ("CG",1))

    x_full = SpatialCoordinate(mesh_full)

    #define intial condition
    def initial_condition(x_full):
        return np.exp(-1000*x_full[1]) # decays exponentially fast away from bottom wall

    #interpolate intial conditions
    u_n_full = Function(V_full_t)
    u_n_full.name = "u_n_full"
    u_n_full.interpolate(initial_condition)



    #define boundary conditions to be the same as before
    outer_below_bc_full = dirichletbc(default_scalar_type(1),locate_dofs_topological(V_full_t, tdim-1, ft_full.find(bottom_wall_marker)),V_full_t)
    outer_above_bc_full = dirichletbc(default_scalar_type(0), locate_dofs_topological(V_full_t, tdim-1, ft_full.find(top_wall_marker)),V_full_t)
    bcs_full = [outer_below_bc_full, outer_above_bc_full]


    #update solution value to match initial condition
    uh_full = Function(V_full_t)
    uh_full.name = "uh_full"
    uh_full.interpolate(initial_condition)






    #define trial test and test functions
    u, v = TrialFunction(V_full_t), TestFunction(V_full_t)
    f_full = Constant(mesh_full, PETSc.ScalarType(0))  #no source term currently so just have f= 0
    a_full = u * v * ufl.dx + delt * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    l_full = (u_n_full + delt * f_full) * v * ufl.dx

    bilinear_form_full = form(a_full)
    linear_form_full = form(l_full)

    A_full = assemble_matrix(bilinear_form_full, bcs_full) #assembles matrix associated with the bilieanr form
    A_full.assemble()
    b_full = create_vector(linear_form_full) # initilaises right hand side based on linear form L (current solution u_n is initial condition)



    solver_full = PETSc.KSP().create(mesh_full.comm)
    # solver_full.setType(PETSc.KSP.Type.CG)
    solver_full.setType(PETSc.KSP.Type.GMRES)

    solver_full.getPC().setType(PETSc.PC.Type.LU)
    solver_full.setOperators(A_full)


    # Set solver options
    solver_full.setTolerances(rtol=1e-6, atol=1e-10, max_it=1000)

    #for gif generation
    # import matplotlib as mpl

    # grid_full = pyvista.UnstructuredGrid(*plot.vtk_mesh(V_full_t))

    # plotter_full = pyvista.Plotter()
    # pyvista.start_xvfb()
    # plotter_full.open_gif("u_full_time_flat_iterativev2{}.gif".format(P), fps=10) #records a gif of solution evolving



    # grid_full.point_data["uh_full"] = uh_full.x.array #makes current solution input for the plotter
    # #warped = grid.warp_by_scalar("uh", factor=0)

    # viridis = mpl.colormaps.get_cmap("viridis")
    # sargs = dict(title = 'c', title_font_size=30, label_font_size=25, color="black",
    #             position_x=0.15, position_y=0.01, width=0.7, height=0.2)

    # # renderer = plotter.add_mesh(warped, show_edges=False, lighting=False,
    # #                             cmap=viridis, scalar_bar_args=sargs,
    # #                             clim=[0, max(uh.x.array)])

    # renderer_full = plotter_full.add_mesh(grid_full, show_edges=False, lighting=False,
    #                             cmap=viridis, scalar_bar_args=sargs,
    #                             clim=[0, max(uh_full.x.array)])
    # text_actor_full = plotter_full.add_text("0", position = "upper_left", font_size=30, color="black")

    # plotter_full.camera.zoom(1) 
    # plotter_full.view_xy()





    ##################################TIME STEPPING LOOP POINT WANT FULL 
    

    ##empty lists to store errors and times
    times_list_round = []
    full_flux_above_list =[]
    full_flux_below_list =[]



    for i in range(num_steps):
        print(n,t)

       ########FULL SIMULATION CODE

        # Update the right hand side reusing the initial vector
        with b_full.localForm() as loc_b_full:
            loc_b_full.set(0) #reinitialises to 0 before updating
        assemble_vector(b_full, linear_form_full)

        # Apply Dirichlet boundary condition to the vector
        apply_lifting(b_full, [bilinear_form_full], [bcs_full])
        b_full.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b_full, bcs_full)

        # Solve linear problem
        solver_full.solve(b_full, uh_full.vector) #solves A.uh = b storing solution in uh
        uh_full.x.scatter_forward() 

        # Update solution at previous time step (u_n)
        u_n_full.x.array[:] = uh_full.x.array #allows for use in linear form


        # Update plot
        #warped = grid.warp_by_scalar("uh", factor=1)
        #plotter.update_coordinates(warped.points.copy(), render=False)
        # plotter_full.update_scalars(uh_full.x.array, render=False) #updates visualisation
        # plotter_full.remove_actor(text_actor_full)

        # # Add the updated text
        # text_actor_full = plotter_full.add_text(f"{t:.3f}", position = "upper_left", font_size=30, color="black")

        # plotter_full.write_frame()




        ######interpolate full solution onto squished mesh for error calculations
        ##interpolating full solution onto original mesh
        coords_hom = mesh_hom.geometry.x
        values_full_squish = np.zeros(coords_hom.shape[0])
        V_vis_t = functionspace(mesh_hom, ("CG",1))
        u_full_squish = Function(V_vis_t)



        # Create bounding box tree for old mesh

        bb_tree_full = geometry.bb_tree(mesh_full,tdim)


        # Interpolate values
        for i, coord in enumerate(coords_hom):
            x, y, _ = coord  # Extract only x and y components, ignore the third (z) component if present
            if y < R - L- epsilon:
                # Directly use the old function value
                point = np.array([x, y, 0.0])
                cell_candidates = geometry.compute_collisions_points(bb_tree_full, point)
                colliding_cells = geometry.compute_colliding_cells(mesh_full, cell_candidates, point)
                if len(colliding_cells) > 0:
                    closest_cell = colliding_cells[0]
                    values_full_squish[i] = uh_full.eval([point], [closest_cell])[0]
                else:
                    values_full_squish[i] = 0
            elif y > R - L- epsilon:
                # Shift y-coordinate by +2L
                shifted_coord = (x, y +2 * L, 0.0)
                cell_candidates = geometry.compute_collisions_points(bb_tree_full, shifted_coord)
                colliding_cells = geometry.compute_colliding_cells(mesh_full, cell_candidates, shifted_coord)
                if len(colliding_cells) > 0:
                    closest_cell = colliding_cells[0]
                    values_full_squish[i] = uh_full.eval([shifted_coord], [closest_cell])[0]
                else:
                    values_full_squish[i] = 0
            else:
                # Set values to zero or some default if desired
                values_full_squish[i] = 0


        # Assign interpolated values to the new function
        u_full_squish.x.array[:] = values_full_squish



        ####### FLUX FOR ERROR CALCULATION
        



        # Average flux over the line
        line_length_above = assemble_scalar(form(1 * dS_hom(flux_line_above_marker)))
        line_length_below = assemble_scalar(form(1 * dS_hom(flux_line_below_marker)))
        
        # Assuming u_full is a finite element function
        flux_full_squish = -grad(u_full_squish)



        # Restrict flux to one side of the facet
        flux_normal_squish = avg(flux_full_squish[1])

        # Define the integral using the restricted flux
        integral_flux_above_squish = flux_normal_squish * dS_hom(flux_line_above_marker)
        integral_flux_below_squish = flux_normal_squish * dS_hom(flux_line_below_marker)

        # Assemble the integral
        flux_total_above_squish = assemble_scalar(form(integral_flux_above_squish))
        flux_total_below_squish = assemble_scalar(form(integral_flux_below_squish))



        # Average flux over the line
        average_flux_above_squish = flux_total_above_squish/ line_length_above
        average_flux_below_squish = flux_total_below_squish/ line_length_below
        full_flux_above_list.append(average_flux_above_squish)
        full_flux_below_list.append(average_flux_below_squish)


        times_list_round.append(round(t,3))


        #update time
        t += delt
        n += 1
        print(t, times_list[n])


    #closing plotters once time loop is finished

    # plotter_full.close()






    #####reformatting - save lists as np arrays to save as txt files more easily
    times_list_roundlongarray = np.array(times_list_round)
    full_flux_above_array_long = np.array(full_flux_above_list)
    full_flux_below_array_long=np.array(full_flux_below_list)





    ###save as txt files to call into plotting code
    np.savetxt('times_list_long_{}'.format(run),times_list_roundlongarray)
    np.savetxt('full_flux_above_array_long_{}'.format(run), full_flux_above_array_long)
    np.savetxt('full_flux_below_array_long_{}'.format(run), full_flux_below_array_long)


    return






def time_sim_em(eps,n_chunk=20,num_steps=12, final_time=3):
    ########################### MESH FOR EFFECTIVE SIMULATION
    t = 0 # intial time
    T = final_time#final time

    delt = (T-t)/num_steps
    delta = width/n_chunk 
    ## for naming purposes
    if eps==0.25:
        run=1
    elif eps==0.05:
        run=2

    P = 1/((L/eps)+(2*delta/np.pi)*(np.log(1/(8*eps))+1)) # defines permeability in terms of channel geometry 


    n=0 # time step
    ########################## MESH FOR EFFECTIVE SIMULATION
    gdim = 2
    model_rank = 0

    gmsh.initialize()


    proc = MPI.COMM_WORLD.rank
    mesh_comm = MPI.COMM_WORLD


    #define markers for domains
    main_marker = 2
    interface_marker = 3
    top_wall_marker=4
    bottom_wall_marker=5
    bottom_sub_marker= 6
    top_sub_marker = 7
    flux_line_above_marker=10
    flux_line_below_marker=11



    if proc == 0:
        # We create one rectangle for each subdomain
        rec1 = gmsh.model.occ.addRectangle(0, 0, 0, width, R-L, tag=1)
        rec2 = gmsh.model.occ.addRectangle(0, R-L, 0, width, R-L, tag=2)
        # We fuse the two rectangles and keep the interface between them - mark as one domain but want interface line?
        gmsh.model.occ.fragment([(2, rec1)], [(2, rec2)])
        gmsh.model.occ.synchronize()
        

        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)
        # Add points for horizontal lines to calculate flux over
        p1_above = gmsh.model.occ.addPoint(0, R - L + dist, 0)
        p2_above = gmsh.model.occ.addPoint(width, R - L + dist, 0)
        line_above = gmsh.model.occ.addLine(p1_above, p2_above)
        gmsh.model.occ.synchronize()


        p1_below = gmsh.model.occ.addPoint(0, R - L - dist, 0)
        p2_below = gmsh.model.occ.addPoint(width, R - L - dist, 0)
        line_below = gmsh.model.occ.addLine(p1_below, p2_below)
        
        
        # Fragment the added lines with the existing geometry
        gmsh.model.occ.fragment([(1, line_above), (1, line_below)], gmsh.model.getEntities(dim=2))
        
        gmsh.model.occ.synchronize()

        # Synchronize and mesh
        gmsh.model.occ.synchronize()
        

        #set mesh size for all points
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)

        #mark different boundaries
        top_wall=[]
        bottom_wall=[]
        side_walls=[]
        interface = []
        flux_line_above_hom =[]
        flux_line_below_hom =[]
        for line in gmsh.model.getEntities(dim=1):
            com = gmsh.model.occ.getCenterOfMass(line[0], line[1])
            if np.isclose(com[1],0):
                bottom_wall.append(line[1])
            elif np.isclose(com[1],2*R-2*L):
                top_wall.append(line[1])
            elif np.isclose(com[1],R-L):
                interface.append(line[1])
            elif np.isclose(com[1], (R-L-dist)/2):
                side_walls.append(line[1])
            elif np.isclose(com[1], (3*(R-L)+dist)/2):
                side_walls.append(line[1])
            elif np.isclose(com[1],R-L-dist/2):
                side_walls.append(line[1])
            elif np.isclose(com[1], R-L+dist/2):
                side_walls.append(line[1])
            elif np.isclose(com[1],R-L-dist):
                flux_line_below_hom.append(line[1])
            elif np.isclose(com[1],R-L + dist):
                flux_line_above_hom.append(line[1])
                

        #mark subdomains
        
        
        bottom_half= []
        top_half =[]
        for surface in gmsh.model.getEntities(dim=2):
            com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
            if np.allclose(com, [width/2, (R-L-dist)/2, 0]):
                bottom_half.append(surface[1])
            elif np.allclose(com, [width/2, R-L-dist/2,0]):
                bottom_half.append(surface[1])
            elif np.allclose(com, [width/2, (3*(R-L)+dist)/2, 0]):
                top_half.append(surface[1]) 
            elif np.allclose(com, [width/2, R-L +dist/2,0]):
                top_half.append(surface[1])
        
        # Synchronize and mesh
        gmsh.model.occ.synchronize()   
            

        gmsh.model.addPhysicalGroup(2, top_half, top_sub_marker)
        gmsh.model.addPhysicalGroup(2, bottom_half, bottom_sub_marker)
        gmsh.model.addPhysicalGroup(1, top_wall, top_wall_marker)
        gmsh.model.addPhysicalGroup(1, bottom_wall, bottom_wall_marker)
        gmsh.model.addPhysicalGroup(1, interface, interface_marker)
        gmsh.model.addPhysicalGroup(1, flux_line_above_hom, flux_line_above_marker)
        gmsh.model.addPhysicalGroup(1, flux_line_below_hom, flux_line_below_marker)
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



    MPI.COMM_WORLD.barrier()


    tdim = mesh_hom.topology.dim #topological dimension of domain

    #define integration measures
    dx_hom = Measure("dx", domain = mesh_hom, subdomain_data= ct_hom)
    dS_hom = Measure("dS", mesh_hom, subdomain_data = ft_hom) 


    ###setting up effective problem

    #MIXED ELEMENT FORMULATION
    #define function space and mixed elements
    V1 = functionspace(mesh_hom, ("CG", 1))
    V2 = functionspace(mesh_hom, ("CG", 1))

    #define mixed function space
    V_hom_t = functionspace(mesh_hom, V1.ufl_element() * V2.ufl_element()) #mixed function space
    V_nm_t = functionspace(mesh_hom, ("CG",1)) #non mixed function space




    x = SpatialCoordinate(mesh_hom)
    #MIXED FUNCTION SPACE BOUNDARY CONDITIONS AND ASSEMBLY
    #define boundary condition
    outer_below_bc = dirichletbc(default_scalar_type(1),locate_dofs_topological(V_hom_t.sub(0), tdim-1, ft_hom.find(bottom_wall_marker)),V_hom_t.sub(0)) 
    outer_above_bc = dirichletbc(default_scalar_type(0), locate_dofs_topological(V_hom_t.sub(1), tdim-1, ft_hom.find(top_wall_marker)),V_hom_t.sub(1))

    bc = [outer_below_bc, outer_above_bc]

    coords = mesh_hom.geometry.x


    #define intial condition
    def initial_condition_1(x):
        return np.exp(-1000*x[1])

    def initial_condition_2(x):
        return 0*x[1]



    ##interpolate initial conditions onto EM functions
    u_EM_0 = Function(V_hom_t)
    u_EM_0.sub(0).interpolate(initial_condition_1)
    u_EM_0.sub(1).interpolate(initial_condition_2)


    chi = ((2*delta*eps)/pi)*(log(1/(8*eps))+1)


    #initial Euler-Maclaurin values
    u_EM1_0,u_EM2_0 = u_EM_0.split()
    u_EM1_init = Function(V_nm_t)
    u_EM2_init = Function(V_nm_t)
    u_EM1_init.interpolate(u_EM1_0)
    u_EM2_init.interpolate(u_EM2_0)


    times_list=[num*delt for num in range(num_steps+1)]
    expvalues =[exp(time) for time in times_list]

    #EM lists to store flux and function values for quadrature
    u_EM1_list=[u_EM1_init] 
    u_EM2_list=[u_EM2_init]
    du_EM1 = Function(V_nm_t)
    du_EM2 = Function(V_nm_t)
    du_EM1.x.array[:] = 0.0
    du_EM2.x.array[:] = 0.0
    du_EM1_list=[du_EM1]
    du_EM2_list=[du_EM2]


    #interpolate intial condition to current and working u
    u_EMn = Function(V_hom_t)
    u_EMn.interpolate(u_EM_0)
    uh_EM = Function(V_hom_t)
    uh_EM.interpolate(u_EM_0)
    u_1_EMn, u_2_EMn = u_EMn.split() #split current solution into its values in each part of the domain




    ##define trial and test functions for EM problem
    (u_EM_1,u_EM_2)= TrialFunctions(V_hom_t)
    (w_EM_1,w_EM_2)= TestFunctions(V_hom_t)


    #### euler maclaurin


    hem_list=[]
    fem_list=[]

    Hemm2 = Function(V_nm_t)
    Hemm2.x.array[:]= 0.0
    hem_list.append(Hemm2)


    Hemm1 = Function(V_nm_t)
    Hemm1.x.array[:]= 0.0
    hem_list.append(Hemm1)

    Femm2 = Function(V_nm_t)
    Femm2.x.array[:]= 0.0
    fem_list.append(Femm2)

    Femm1 = Function(V_nm_t)
    Femm1.x.array[:]= 0.0
    fem_list.append(Femm1)


    #for EM
    p = Constant(mesh_hom, PETSc.ScalarType(P))
    jmp_EM = avg(u_EM_2) - avg(u_EM_1)
    jmp_w = avg(w_EM_2)- avg(w_EM_1)
    plus_w = avg(w_EM_1) + avg(w_EM_2)

    a_EM =(u_EM_1*w_EM_1 + delt*inner(grad(u_EM_1), grad(w_EM_1)))*dx_hom(bottom_sub_marker) + (u_EM_2*w_EM_2 + delt*inner(grad(u_EM_2),grad(w_EM_2)))*dx_hom(top_sub_marker) + delt*p*jmp_w*jmp_EM*dS_hom(interface_marker)

    #interfaceconditions


    def EM_f_term(t,n):
        return sqrt(t/pi)*((2*eps*L**2)/(L+chi))*(fem_list[n+2].x.array-fem_list[n+1].x.array)

    def EM_h_term(t,n):
        return sqrt(t/pi)*2*eps*L*(hem_list[n+2].x.array - hem_list[n+1].x.array)

    EMh = Function(V_nm_t)
    EMf = Function(V_nm_t)



    ######## assemble linear and bilinear form for Euler-Maclaurin
    bilinear_form_EM = form(a_EM)
    A_Em= assemble_matrix(bilinear_form_EM,bc)
    A_Em.assemble()


    l_EM = u_1_EMn*w_EM_1*dx_hom(bottom_sub_marker) + u_2_EMn*w_EM_2*dx_hom(top_sub_marker) + (-EMf*jmp_w - EMh*plus_w)*dS_hom(interface_marker)

    linear_form_EM = form(l_EM)
    b_EM = create_vector(linear_form_EM)

    #define linear form within each time step


    # Create solver

    solver_EM = PETSc.KSP().create(mesh_hom.comm)

    # Set solver type 
    solver_EM.setType(PETSc.KSP.Type.GMRES)

    # Set preconditioner to LU

    pc_EM = solver_EM.getPC()
    pc_EM.setType(PETSc.PC.Type.LU)

    # Set operator matrix
    A_Em.shift(1e-10)  # Improves invertibility
    solver_EM.setOperators(A_Em)

    # Set solver options
    solver_EM.setTolerances(rtol=1e-6, atol=1e-10, max_it=1000)



    # #import for plotting ang get plotter ready
    # import matplotlib as mpl

    #define new functionspace for visualisation
    #interpolate to solve
    V_vis_t = functionspace(mesh_hom, ("CG",1))

    u_EM_1_vis = Function(V_vis_t)
    u_EM_2_vis = Function(V_vis_t)



    ##for euler maclaurin
    u1_EM_sol, u2_EM_sol= uh_EM.split()
    u_EM_1_vis.interpolate(u1_EM_sol)
    u_EM_2_vis.interpolate(u2_EM_sol)
    u_EM_vis = Function(V_vis_t)
    du_EM_1_sol = Function(V_vis_t)
    du_EM_2_sol = Function(V_vis_t)


    u1_EM_vis_array = u_EM_1_vis.x.array
    u2_EM_vis_array =u_EM_2_vis.x.array


    coords = mesh_hom.geometry.x
    y_coords = coords[:, 1]  # y-coordinate of the mesh vertices

    # Define a threshold to identify the upper half of the domain
    threshold = R - L - epsilon

    # Find indices where y-coordinates are greater than the threshold
    indices1 = np.where(y_coords > threshold)[0]
    indices2 = np.where(y_coords < threshold)[0]



    # Set the corresponding solution values to zero

    u1_EM_vis_array[indices1] = 0
    u2_EM_vis_array[indices2] = 0


    u_EM_vis.vector[:] = u1_EM_vis_array + u2_EM_vis_array



    # ##plotter for EM
    # grid_EM = pyvista.UnstructuredGrid(*plot.vtk_mesh(V_vis_t))

    # plotter_EM = pyvista.Plotter()
    # pyvista.start_xvfb()
    # plotter_EM.open_gif("u_EM_time_flat_iterativev2{}.gif".format(P), fps=10) #records a gif of solution evolving
    # grid_EM.point_data["u_EM_vis"] = u_EM_vis.x.array #makes current solution input for the plotter


    # viridis = mpl.colormaps.get_cmap("viridis")
    # sargs = dict(title = 'c', title_font_size=30, label_font_size=25, color="black",
    #             position_x=0.15, position_y=0.01, width=0.7, height=0.2)


    # renderer = plotter_EM.add_mesh(grid_EM, show_edges=False, lighting=False,
    #                             cmap=viridis, scalar_bar_args=sargs,
    #                             clim=[0, max(u_EM_vis.x.array)])

    # text_actor_EM = plotter_EM.add_text("0", position = "upper_left", font_size=30, color="black")

    # plotter_EM.camera.zoom(1) 
    # plotter_EM.view_xy()


    # #define new mesh and functionspace to have gap for effective problem
    # gap_mesh = create_rectangle(comm, [[0,0],[width, 2*R]],[200,200])

    # #create functionspace on new mesh
    # V_gap = functionspace(gap_mesh, ("CG",1))

    # #access coordinates of new mesh
    # coords_gap = gap_mesh.geometry.x

    # #interpolate solutions onto mesh
    # #access coordinates of old mesh
    # coords_hom = mesh_hom.geometry.x

    # #create emtpy array to store function values in

    # values_gap_EM = np.zeros(coords_gap.shape[0])


    # #create function in new functionspace to interpolate solution onto

    # u_gap_EM = Function(V_gap)

    # #create bounding box tree for old mesh to allow for quick searching
    # bb_tree_hom = geometry.bb_tree(mesh_hom, tdim)

    # #create emtpy array to store function values in
    # values_gap_hom = np.zeros(coords_gap.shape[0])



    # #create function in new functionspace to interpolate solution onto
    # u_gap_hom = Function(V_gap)


    # #create bounding box tree for old mesh to allow for quick searching
    # bb_tree_hom = geometry.bb_tree(mesh_hom, tdim)

    # #loop over each point to interpolate values
    # for i, coord in enumerate(coords_gap):
    #     x,y,_ = coord
    #     if y < R-L: #for y values below membrane can use original values as is
    #         point = np.array([x,y,0])
    #         cell_candidates = geometry.compute_collisions_points(bb_tree_hom,point)
    #         colliding_cells = geometry.compute_colliding_cells(mesh_hom, cell_candidates, point)
    #         if len(colliding_cells)>0:
    #             closest_cell = colliding_cells[0]
    #             values_gap_hom[i] = u_vis.eval([point],[closest_cell])[0]
    #             values_gap_EM[i] = u_EM_vis.eval([point],[closest_cell])[0]
    #         else:
    #             values_gap_hom[i]=0
    #             values_gap_EM[i]=0
    #     elif y> R + L:
    #         shifted_coord = (x, y-2*L,0)
    #         cell_candidates = geometry.compute_collisions_points(bb_tree_hom,shifted_coord)
    #         colliding_cells = geometry.compute_colliding_cells(mesh_hom,cell_candidates,shifted_coord)
    #         if len(colliding_cells)>0:
    #             closest_cell= colliding_cells[0]
    #             values_gap_hom[i] = u_vis.eval([shifted_coord],[closest_cell])[0]
    #             values_gap_EM[i] = u_EM_vis.eval([shifted_coord],[closest_cell])[0]
    #         else:
    #             values_gap_hom[i]=0
    #             values_gap_EM[i]=0
    #     else:
    #         values_gap_hom[i]=0
    #         values_gap_EM[i]=0


    # u_gap_hom.x.array[:] =values_gap_hom

    # u_gap_EM.x.array[:] =values_gap_EM

    # grid_gap = pyvista.UnstructuredGrid(*plot.vtk_mesh(V_gap))

    # plotter_gap= pyvista.Plotter()
    # pyvista.start_xvfb()
    # plotter_gap.open_gif("u_hom_time_gap_iterativev2{}.gif".format(P), fps=10) #records a gif of solution evolving


    # grid_gap.point_data["u_gap_hom"] = u_gap_hom.x.array #makes current solution input for the plotter


    # viridis = mpl.colormaps.get_cmap("viridis")
    # sargs = dict(title = 'c', title_font_size=30, label_font_size=25, color="black",
    #             position_x=0.15, position_y=0.01, width=0.7, height=0.2)



    # renderer_gap = plotter_gap.add_mesh(grid_gap, show_edges=False, lighting=False,
    #                             cmap=viridis, scalar_bar_args=sargs,
    #                             clim=[0, max(u_vis.x.array)])


    # text_actor_gap = plotter_gap.add_text("0", position = "upper_left", font_size=30, color="black")


    # plotter_gap.camera.zoom(1) 
    # plotter_gap.view_xy()






    # grid_gapEM = pyvista.UnstructuredGrid(*plot.vtk_mesh(V_gap))


    # ####gap plotter for EM
    # plotter_gap_EM= pyvista.Plotter()
    # pyvista.start_xvfb()
    # plotter_gap_EM.open_gif("u_EM_time_gap_iterativev2{}.gif".format(P), fps=10) #records a gif of solution evolving


    # grid_gapEM.point_data["u_gap_EM"] = u_gap_EM.x.array #makes current solution input for the plotter

    # viridis = mpl.colormaps.get_cmap("viridis")
    # sargs = dict(title = 'c', title_font_size=30, label_font_size=25, color="black",
    #             position_x=0.15, position_y=0.01, width=0.7, height=0.2)

    # #

    # renderer_gap_EM = plotter_gap_EM.add_mesh(grid_gapEM, show_edges=False, lighting=False,
    #                             cmap=viridis, scalar_bar_args=sargs,
    #                             clim=[0, max(u_gap_EM.x.array)])


    # text_actor_gap_EM = plotter_gap_EM.add_text("0", position = "upper_left", font_size=30, color="black")


    # plotter_gap_EM.camera.zoom(1) 
    # plotter_gap_EM.view_xy()






    ################################## TIME STEPPING LOOP ###################################


    ##empty lists to store errors and times
    times_list_round = []


    EM_flux_above_list=[]
    EM_flux_below_list=[]






    for i in range(num_steps):
        print(n,t)



        ### euler maclaurin code
        hemn = Function(V_nm_t)
        hemn.x.array[:]= (u_EM2_list[n].x.array + u_EM1_list[n].x.array)/L - chi/(2*eps)*(du_EM2_list[n].x.array - du_EM1_list[n].x.array)
        femn= Function(V_nm_t)
        femn.x.array[:] = (u_EM2_list[n].x.array - u_EM1_list[n].x.array)/L - chi/(2*eps)*(du_EM2_list[n].x.array + du_EM1_list[n].x.array)
        hem_list.append(hemn)
        fem_list.append(femn)
        
        
        EMh.x.array[:]= EM_h_term(t,n)[:]
        EMf.x.array[:] = EM_f_term(t,n)[:]  

        #Update the right hand side reusing the initial vector
        with b_EM.localForm() as loc_b_EM:
            loc_b_EM.set(0) #reinitialises to 0 before updating
        assemble_vector(b_EM, linear_form_EM)
        # Apply Dirichlet boundary condition to the vector
        apply_lifting(b_EM, [bilinear_form_EM], [bc])
        b_EM.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b_EM, bc)
        
        # Solve linear problem
        solver_EM.solve(b_EM, uh_EM.vector) #solves A.uh = b storing solution in uh
        uh_EM.x.scatter_forward() 

        
        
        # Update solution at previous time step (u_n)
        u_EMn.x.array[:] = uh_EM.x.array #allows for use in linear form


        aEM,gEM = Function(V_hom_t)
        aEM,gEM = uh_EM.split()
        u1_EM_sol.x.array[:], u2_EM_sol.x.array[:]= aEM.x.array, gEM.x.array

        #EM
        u_EM_1_vis.interpolate(u1_EM_sol)
        u_EM_2_vis.interpolate(u2_EM_sol)


        u1_EM_vis_array = u_EM_1_vis.x.array
        u2_EM_vis_array =u_EM_2_vis.x.array

        #EM
        u1_EM_vis_array[indices1] = 0
        u2_EM_vis_array[indices2] = 0
        u_EM_1_vis.x.array[indices1]=0
        u_EM_2_vis.x.array[indices2]=0

        
        du_EM_1_sol.x.array[:] = eps/(L+chi)*(u1_EM_vis_array - u2_EM_vis_array) -EM_h_term(t,n)[:] + EM_f_term(t,n)[:]
        du_EM_2_sol.x.array[:] = eps/(L+chi)*(u1_EM_vis_array - u2_EM_vis_array) +EM_h_term(t,n)[:] + EM_f_term(t,n)[:]

        #EM
        u_EM1_list.append(u_EM_1_vis)
        u_EM2_list.append(u_EM_2_vis)
        du_EM1_list.append(du_EM_1_sol)
        du_EM2_list.append(du_EM_2_sol)

        #EM
        u_EM_vis.x.array[:] = u1_EM_vis_array + u2_EM_vis_array

        # updating gif
        # plotter_EM.update_scalars(u_EM_vis.x.array, render=False) #updates visualisation
        # plotter_EM.remove_actor(text_actor_EM)


        # text_actor_EM = plotter_EM.add_text(f"{t:.3f}", position = "upper_left", font_size=30, color="black")

        # plotter_EM.write_frame()

        # print("linear form interface=",assemble_scalar(form((-avg(EMf)*jmp_w - avg(EMh)*plus_w)*dS_hom(interface_marker))))

        


        # ###to visualise on gap!

        # #loop over each point to interpolate values
        # for i, coord in enumerate(coords_gap):
        #     x,y,_ = coord
        #     if y < R-L: #for y values below membrane can use original values as is
        #         point = np.array([x,y,0])
        #         cell_candidates = geometry.compute_collisions_points(bb_tree_hom,point)
        #         colliding_cells = geometry.compute_colliding_cells(mesh_hom, cell_candidates, point)
        #         if len(colliding_cells)>0:
        #             closest_cell = colliding_cells[0]
        #             values_gap_hom[i] = u_vis.eval([point],[closest_cell])[0]
        #             values_gap_EM[i] = u_EM_vis.eval([point],[closest_cell])[0]
        #         else:
        #             values_gap_hom[i]=0
        #             values_gap_EM[i]=0
        #     elif y> R + L:
        #         shifted_coord = (x, y-2*L,0)
        #         cell_candidates = geometry.compute_collisions_points(bb_tree_hom,shifted_coord)
        #         colliding_cells = geometry.compute_colliding_cells(mesh_hom,cell_candidates,shifted_coord)
        #         if len(colliding_cells)>0:
        #             closest_cell= colliding_cells[0]
        #             values_gap_hom[i] = u_vis.eval([shifted_coord],[closest_cell])[0]
        #             values_gap_EM[i] = u_EM_vis.eval([shifted_coord],[closest_cell])[0]
        #         else:
        #             values_gap_hom[i]=0
        #             values_gap_EM[i]=0
        #     else:
        #         values_gap_hom[i]=0
        #         values_gap_EM[i]=0

        # u_gap_EM.x.array[:] =values_gap_EM

        # u_gap_hom.x.array[:] =values_gap_hom


        
        # plotter_gap.update_scalars(u_gap_hom.x.array, render=False) #updates visualisation
        # # Update the title
        # # Remove the old text
        # plotter_gap.remove_actor(text_actor_gap)

        # # Add the updated text
        # text_actor_gap = plotter_gap.add_text(f"{t:.3f}", position = "upper_left", font_size=30, color="black")
        # plotter_gap.write_frame()

        

        # #EM
        # plotter_gap_EM.update_scalars(u_gap_EM.x.array, render=False) #updates visualisation
        # # Update the title
        # # Remove the old text
        # plotter_gap_EM.remove_actor(text_actor_gap_EM)

        # # Add the updated text
        # text_actor_gap_EM = plotter_gap_EM.add_text(f"{t:.3f}", position = "upper_left", font_size=30, color="black")
        # plotter_gap_EM.write_frame()




        

        ####### FLUX FOR ERROR CALCULATION

        #NUMERICAL SOLUTION OF EFFECTIVE

        flux_EM = -grad(u_EM_vis)


        # Restrict flux to one side of the facet

        flux_normal_EM = avg(flux_EM[1])

        # Define the integral using the restricted flux

        integral_flux_above_EM = flux_normal_EM * dS_hom(flux_line_above_marker)
        integral_flux_below_EM = flux_normal_EM * dS_hom(flux_line_below_marker)

        

        # Assemble the integral


        flux_total_above_EM = assemble_scalar(form(integral_flux_above_EM))
        flux_total_below_EM = assemble_scalar(form(integral_flux_below_EM))




        # Average flux over the line
        line_length_above = assemble_scalar(form(1 * dS_hom(flux_line_above_marker)))
        line_length_below = assemble_scalar(form(1 * dS_hom(flux_line_below_marker)))


        #EM
        average_flux_above_EM = flux_total_above_EM / line_length_above
        average_flux_below_EM = flux_total_below_EM / line_length_below
        EM_flux_above_list.append(average_flux_above_EM)
        EM_flux_below_list.append(average_flux_below_EM)




        
        

        times_list_round.append(round(t,3))


        #update time
        t += delt
        n += 1
        print(t, times_list[n])


    # plotter_EM.close()
    # plotter_gap_EM.close()






    #####reformatting - save lists as np arrays to save as txt files more easily
    times_list_roundshortarray = np.array(times_list_round)
    EM_flux_above_array = np.array(EM_flux_above_list)
    EM_flux_below_array = np.array(EM_flux_below_list)




    ###save as txt files to call into plotting code
    np.savetxt("times_list_short_{}".format(run),times_list_roundshortarray)
    np.savetxt('EM_flux_above_array_{}'.format(run), EM_flux_above_array)
    np.savetxt('EM_flux_below_array_{}'.format(run), EM_flux_below_array)


eps_list =[0.25,0.05]
for eps in eps_list:
    time_sim_effective(eps)
    time_sim_full(eps)
    time_sim_em(eps)
