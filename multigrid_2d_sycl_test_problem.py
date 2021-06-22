import dolfinx
import numpy as np
import ufl
import scipy as scp
from scipy import sparse
from mpi4py import MPI
from petsc4py import PETSc
import multigrid_solver as ms
import csv

num_levels = 5
mu0 = 1
mu1 = 3
mu2 = 3
omega = 4/5
mesh_finest = None
a_finest = None
L_finest = None
bcs_finest = None
b_finest = None
V_finest = None
num_elems_coarsest = 16
finest_level = num_levels
coarsest_level = 1

current_mesh = dolfinx.UnitSquareMesh(
    MPI.COMM_WORLD, num_elems_coarsest, num_elems_coarsest, dolfinx.cpp.mesh.CellType.triangle)

# Store the ProblemVar's properties
topology_to_space_list = np.empty((finest_level-coarsest_level+1,1),dtype=np.ndarray)
b_list = np.empty((finest_level-coarsest_level+1,1), dtype=np.ndarray)
A_sp_list = np.empty((finest_level-coarsest_level+1,1), dtype=object)
A_jacobi_sp_list = np.empty((finest_level-coarsest_level+1,1), dtype=object)
parent_info_vertex_list = np.empty((finest_level-coarsest_level,1), dtype=np.ndarray)
paren_info_edges_list = np.empty((finest_level-coarsest_level,1), dtype=np.ndarray)
coarse_edges_list = np.empty((finest_level-coarsest_level,1), dtype=np.ndarray)
coarsest_level_Eigen_matrix = None

# Function to map topology to space dofs
def topology_to_space_map(mesh, space , num_dofs):
    cells = mesh.topology.connectivity(2, 0)
    topology_to_space_mapping = np.empty((num_dofs,1))
    for c in range(cells.num_nodes):
        cur_cell = cells.links(c)
        for v, dof in zip(cur_cell, space.dofmap.cell_dofs(c)):
            topology_to_space_mapping[v] = dof
    return topology_to_space_mapping

#Function to convert parent info to required SYCL format
def parent_info_convertor(parent_map , num_dofs_parent, num_dofs_child):
    parent_info_vertex = np.empty((num_dofs_parent,1))
    parent_info_edges = np.empty((num_dofs_child,1))
    for i in range(0, num_dofs_parent):
        parent_info_vertex[i] = parent_map[i][1]
    for i in range(num_dofs_parent, num_dofs_child):
        parent_info_edges[i - num_dofs_parent] = parent_map[i][1]
    return parent_info_vertex , parent_info_edges



def coarse_edges_to_array(edges):
    num_edges = edges.num_nodes
    coarse_edges_list = np.empty((2*num_edges,1))
    for i in range(0,num_edges):
        coarse_edges_list[i] = edges.links(i)[0]
        coarse_edges_list[i+1] = edges.links(i)[1]
    return coarse_edges_list

def getJacobiMatrices(A_mat):
    # Takes in a Global Stiffness Matrix in scipy sparse and gives the Final Jacobi Iteration Matrices
    A_mat_diag = A_mat.diagonal()
    R_mat = A_mat - scp.sparse.diags(A_mat_diag, 0)
    A_mat_dig_inv = 1 / A_mat_diag
    diag_A_inv = scp.sparse.diags(A_mat_dig_inv, 0)
    R_omega_mat = diag_A_inv.dot(R_mat)
    return R_omega_mat, diag_A_inv

# Iterating over the levels
for i in range(1, num_levels+1):
    current_mesh.topology.create_entities(1)
    V_i = dolfinx.FunctionSpace(current_mesh, ("CG", 1))
    num_dofs_current = current_mesh.topology.connectivity(0,0).num_nodes
    topology_to_space_list[i] = topology_to_space_map(current_mesh, V_i , num_dofs_current)
    #print(current_mesh.topology.connectivity(1, 0))
    if i is not num_levels:
        coarse_edges_list[i] = coarse_edges_to_array(current_mesh.topology.connectivity(1, 0))
    uD_i = dolfinx.Function(V_i)
    uD_i.interpolate(lambda x: 1+x[0]**2 + 2*x[1]**2)
    uD_i.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                            mode=PETSc.ScatterMode.FORWARD)
    fdim_i = current_mesh.topology.dim-1
    current_mesh.topology.create_connectivity(
        fdim_i, current_mesh.topology.dim)
    boundary_facets_i = np.where(np.array(
        dolfinx.cpp.mesh.compute_boundary_facets(current_mesh.topology)) == 1)[0]
    boundary_dofs_i = dolfinx.fem.locate_dofs_topological(
        V_i, fdim_i, boundary_facets_i)
    bc_i = dolfinx.DirichletBC(uD_i, boundary_dofs_i)
    u_i = ufl.TrialFunction(V_i)
    v_i = ufl.TestFunction(V_i)
    f_i = dolfinx.Constant(current_mesh, -6)
    a_i = ufl.dot(ufl.grad(u_i), ufl.grad(v_i)) * ufl.dx
    A_i = dolfinx.fem.assemble_matrix(a_i, bcs=[bc_i])
    A_i.assemble()

    assert isinstance(A_i, PETSc.Mat)
    ai, aj, av = A_i.getValuesCSR()
    del A_i
    if i is not coarsest_level:
        A_sp_list[i] = ms.csr_matrix_elements(ai , aj , av , num_dofs_current)
    else:
        coarsest_level_Eigen_matrix = ms.eigen_matrix_assemble(ai , aj , av , num_dofs_current)

    A_sp_i = scp.sparse.csr_matrix((av, aj, ai))
    del av, ai, aj
    R_mat_i , dinv_i = getJacobiMatrices(A_sp_i)
    del A_sp_i
    R_mat_i_sycl = ms.csr_matrix_elements(R_mat_i.indptr, R_mat_i.indices , R_mat_i.data, num_dofs_current)
    dinv_i_sycl = ms.csr_matrix_elements(dinv_i.indptr, dinv_i.indices , dinv_i.data, num_dofs_current)
    A_jacobi_sp_list[i] = ms.csr_jacobi_elements(dinv_i_sycl , R_mat_i_sycl)

    L_i = f_i * v_i * ufl.dx
    b_i = dolfinx.fem.create_vector(L_i)
    with b_i.localForm() as loc_b:
        loc_b.set(0)
    dolfinx.fem.assemble_vector(b_i, L_i)
    dolfinx.fem.apply_lifting(b_i, [a_i], [[bc_i]])
    b_i.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                    mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.set_bc(b_i, [bc_i])
    print(type(b_i.array))

    b_list[i] = np.array(b_i.array).reshape(
        ((num_elems_coarsest)*2**(i-1) + 1) ** 2, 1)
    if i == num_levels:
        mesh_finest = current_mesh
        L_finest = L_i
        a_finest = a_i
        bcs_finest = bc_i
        V_finest = V_i
        b_finest = b_i
    else:
        current_mesh, parent_info = dolfinx.mesh.refine(current_mesh)
        num_dofs_child = current_mesh.topology.connectivity(0,0).num_nodes
        parent_info_vertex_list[i+1] , paren_info_edges_list[i+1] = parent_info_convertor(parent_info.parent_map, num_dofs_current,num_dofs_child)

# Solving the Dolfinx CG1 solution
problem_dolfx_CG1 = dolfinx.fem.LinearProblem(a_finest, L_finest, bcs=[bcs_finest], petsc_options={
    "ksp_type": "preonly", "pc_type": "lu"})
uh_dolfx_CG1 = problem_dolfx_CG1.solve()

# Generating the exact CG2 solution
V2_fine = dolfinx.FunctionSpace(mesh_finest, ("CG", 2))
u_exact_fine = dolfinx.Function(V2_fine)
u_exact_fine.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)
u_exact_fine.vector.ghostUpdate(
    addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
L2_error_dolfx = ufl.inner(uh_dolfx_CG1 - u_exact_fine,
                           uh_dolfx_CG1 - u_exact_fine) * ufl.dx
error_L2_dolfx_norm = np.sqrt(dolfinx.fem.assemble_scalar(L2_error_dolfx))

problem_var = ms.ProblemVar(coarsest_level_Eigen_matrix , A_sp_list , A_jacobi_sp_list,topology_to_space_list,b_list,\
    parent_info_vertex_list, paren_info_edges_list,coarse_edges_list)


u_FMG_sycl = ms.solve(problem_var)

