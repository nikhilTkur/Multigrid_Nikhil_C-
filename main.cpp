#include "poisson.h"
#include <cmath>
#include <dolfinx.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/petsc.h>
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>
#include <dolfinx/refinement/refine.h>
#include "petscmat.h"
#include <vector>
#include "Multigrid_functions.h"
#include <CL/sycl.hpp>

using namespace dolfinx;

int main(int argc, char* argv[])
{
    ProblemVar obj;
    int num_levels = 5;
    extern const int finest_level;
    common::subsystem::init_logging(argc, argv);
    common::subsystem::init_petsc(argc, argv);

    // starter mesh for the domain. Gets refined at the last step of the following for loop.
    auto mesh = std::make_shared<mesh::Mesh>(generation::RectangleMesh::create(
        MPI_COMM_WORLD, { {{0.0, 0.0, 0.0}, {1.0, 1.0, 0.0}} }, { 16, 16},
        mesh::CellType::triangle, mesh::GhostMode::none));
    {
        // Create mesh and function space
        for (int level = 1; level <= finest_level; level++) {
            // Looping over the number of levels = 5

            mesh.topology_mutable().create_entities(1); // Creating the edges for refinement process
            auto V = fem::create_functionspace(functionspace_form_poisson_a, "u", mesh);
            auto V_dofmap = V.dofmap();

            // GET TOPO TO SPACE DICT FOR THE CURRENT MESH AND STORE IT IN OBJECT PROPERTY
            const std::int32_t num_vertices = mesh.topology().index_map(0)->size_local() 
                + mesh.topology().index_map(0)->num_ghosts();
            std::vector<std::uint32_t> topo_to_space_map(num_vertices); // CREATE A SPECIFIC SIZE VECTOR
            auto cells = mesh.topology().connectivity(2, 0);
            for (int c = 0; c < cells->num_nodes(); ++c){
                auto V_dofs = V_dofmap.links(c); // Space Dofs for current cell
                auto vertices = cells->links(c); // Topology vertices for current cell
                for (std::size_t i = 0; i < vertices.size(); ++i) {
                    topo_to_space_map[vertices[i]] = V_dofs[i];
                }
            }
            //  STORE THE TOPOLOGY TO SPACE MAPPING INTO THE PROGRAM OBJECT PROPERTY BUFFER ON DEVICE
            obj.topo_to_space_dict[level] = cl::sycl::buffer<std::uint32_t, 1>{ topo_to_space_map };

            // GET EDGES AND STORE IT IN OBJECT PROPERTY
            std::vector<std::int32_t> edge_list(2 * num_vertices);
            auto edges = mesh.topology().connectivity(1, 0);
            for (int e = 0; e < edges->num_nodes(); e++) {
                auto edge_dofs = edges->links(e); // Storing the adjacency list as a vector
                edge_list[2 * e] = edge_dofs[0];
                edge_list[2 * e + 1] = edge_dofs[1];
            }
            obj.coarse_grid_edges_dict[level] = cl::sycl::buffer<std::int32_t, 1>{ edge_list };

            auto f = std::make_shared<fem::Function<PetscScalar>>(V);

            // Define variational forms
            auto a = std::make_shared<fem::Form<PetscScalar>>(
                fem::create_form<PetscScalar>(*form_poisson_a, { V, V }, {},{}, {}));

            auto L = std::make_shared<fem::Form<PetscScalar>>(
                fem::create_form<PetscScalar>(*form_poisson_L, { V },
                    { {"f", f} }, {}, {}));

            // BCS TO BE UPDATED AS PER THE NEW PROBLEM

            auto u0 = std::make_shared<fem::Function<PetscScalar>>(V);
            u0->interpolate(
                [](const xt::xtensor<double, 2>& x) -> xt::xarray<PetscScalar>
                {
                    return 1 + xt::square(xt::row(x, 0)) + 2 * xt::square(xt::row(x, 1));
                });

            const auto bdofs = fem::locate_dofs_geometrical(
                  { *V },
                  [](const xt::xtensor<double, 2>& x) -> xt::xtensor<bool, 1>
                  {
                      auto x0 = xt::row(x, 0);
                      auto x1 = xt::row(x, 1);
                      return xt::isclose(x0, 0.0) or xt::isclose(x0, 1.0) or xt::isclose(x1, 0.0) or xt::isclose(x1, 1.0);
                  });

            std::vector bc{ std::make_shared<const fem::DirichletBC<PetscScalar>>(u0, std::move(bdofs)) };

            f->interpolate(
                [](const xt::xtensor<double, 2>& x) -> xt::xarray<PetscScalar>
                {
                    return -6.0;
                });

           /* f->interpolate(
                [](const xt::xtensor<double, 2>& x) -> xt::xarray<PetscScalar>
                {
                    auto dx = xt::square(xt::row(x, 0) - 0.5)
                        + xt::square(xt::row(x, 1) - 0.5);
                    return 10 * xt::exp(-(dx) / 0.02);
                });*/

            fem::Function<PetscScalar> u(V);
            la::PETScMatrix A = la::PETScMatrix(fem::create_matrix(*a), false);
            la::PETScVector b(*L->function_spaces()[0]->dofmap()->index_map,
                L->function_spaces()[0]->dofmap()->index_map_bs());

            MatZeroEntries(A.mat());
            fem::assemble_matrix(la::PETScMatrix::set_block_fn(A.mat(), ADD_VALUES), *a,bc);
            MatAssemblyBegin(A.mat(), MAT_FLUSH_ASSEMBLY);
            MatAssemblyEnd(A.mat(), MAT_FLUSH_ASSEMBLY);
            fem::set_diagonal(la::PETScMatrix::set_fn(A.mat(), INSERT_VALUES), *V, bc);
            MatAssemblyBegin(A.mat(), MAT_FINAL_ASSEMBLY);
            MatAssemblyEnd(A.mat(), MAT_FINAL_ASSEMBLY);

            // GET THE MATRIX AND ASSIGN IT TO THE SYCL HANDLE
            double* mat_vals = nullptr;
            std::int32_t* rows = nullptr;
            std::int32_t* cols = nullptr;
            std::int32_t n_rows;
            MatSeqGetArray(A, &mat_vals);
            PetscBool done = PETSC_FALSE;
            MatSeqGetRowIJ(A, 0, PETSC_TRUE, PETSC_FALSE, &n_rows , &rows , &cols, &done);
            MatInfo info;
            MatGetInfo(A, MAT_LOCAL, &info);
            std::int32_t nnz = info.nz_allocated;

            VecSet(b.vec(), 0.0);
            VecGhostUpdateBegin(b.vec(), INSERT_VALUES, SCATTER_FORWARD);
            VecGhostUpdateEnd(b.vec(), INSERT_VALUES, SCATTER_FORWARD);
            fem::assemble_vector_petsc(b.vec(), *L);
            fem::apply_lifting_petsc(b.vec(), { a }, { {bc} }, {}, 1.0);
            VecGhostUpdateBegin(b.vec(), ADD_VALUES, SCATTER_REVERSE);
            VecGhostUpdateEnd(b.vec(), ADD_VALUES, SCATTER_REVERSE);
            fem::set_bc_petsc(b.vec(), bc, nullptr);
            std::int32_t n_dofs;
            VecGetSize(b, &n_dofs);
            double* b_array = nullptr;
            VecGetArray(b, &b_array);
            std::vector<double> b_vec(b_array, b_array + n_dofs);

            // Store the current level matrix in obj's property
            obj.A_sp_dict[level] = csr_matrix_elements(rows, cols, mat_vals, nnz, n_dofs);

            // Store the current level RHS
            obj.b_dict[level] = cl::sycl::buffer<double, 1>{ b_vec };

            // REFINE THE MESH AND MAKE IT CURRENT MESH.
            std::pair<dolfinx::mesh::Mesh, dolfinx::refinement::ParentRelationshipInfo> refine_mesh_info 
                = dolfinx::refinement::refine(mesh);
            mesh = refine_mesh_info.first;

            //STORE THE PARENT INFO IN THE PROGRAM OBJECT PROPERTY
            std::vector<std::uint32_t> vertex(n_dofs);    // Stores the vertex to vertex map of fine to coarse
            std::vector<std::uint32_t>edges(refine_mesh_info.second.parent_map().size() - n_dofs); // Sotres the vertex to edge map of fine to coarse
            for(int k = 0; k < n_dofs; k++){
                vertex[k] = refine_mesh_info.second.parent_map()[k].second;
            }
            for (int k = n_dofs; k < refine_mesh_info.second.parent_map().size(); k++) {
                edges[k - n_dofs] = refine_mesh_info.second.parent_map()[k].second;
            }
            obj.parent_info_vertex_dict[level] = cl::sycl::buffer<std::uint32_t, 1>{ vertex };
            obj.parent_info_edges_dict[level] = cl::sycl::buffer<std::uint32_t, 1>{ edges };
        }
    }
    solve(obj); // Multigrid Solver 
    common::subsystem::finalize_petsc();

    // Store the multigrid solution as a VTK file and compare. 
    return 0;
}
