#ifndef MULTIGRID_H
#define MULTIGRID_H

#include <CL/sycl.hpp>
#include "oneapi/mkl.hpp"
#include "mkl_types.h"
#include <Eigen/SparseLU>
#include <vector>
#include <unordered_map>

struct csr_matrix_elements
{
	cl::sycl::buffer<int, 1> row;
	cl::sycl::buffer<int, 1> col;
	cl::sycl::buffer<double, 1> values;
	oneapi::mkl::sparse::matrix_handle_t matrix;
	int size;
};

class ProblemVar {
public:
	Eigen::SparseMatrix<double> coarsest_level_matrix;
	std::vector<csr_matrix_elements> A_sp_dict;
	//std::vector<csr_jacobi_elements> A_jacobi_sp_dict;

	//[level -> [index = topo dof , value = space dof]]
	std::unordered_map<int, cl::sycl::buffer<std::uint32_t, 1>> topo_to_space_dict;
	std::unordered_map<int, cl::sycl::buffer<double, 1>> b_dict;
	std::unordered_map<int, std::uint32_t> num_dofs_per_level;

	//[level -> [index = fine dof , value = matching coarse dof]]
	std::unordered_map<int, cl::sycl::buffer<std::uint32_t, 1>>parent_info_vertex_dict;

	//[level -> [index = fine dof - vec_2h_dim , value = matching coarse edge]]
	std::unordered_map<int, cl::sycl::buffer<std::uint32_t, 1>>parent_info_edges_dict;

	//[level -> [2 * index = coarse_grid edge , value = coarse_dofs]] eg [1 -> [1,2,3,4,5,6]] -> edge 0 = 1,2 | edge 1 = 3,4, also, edges index from 0 to no_of_edges
	std::unordered_map<int, cl::sycl::buffer<std::uint32_t, 1>>coarse_grid_edges_dict;

	std::unordered_map<int, cl::sycl::buffer<double, 1>> vecs_dict;
	std::unordered_map<int, cl::sycl::buffer<double, 1>> temp_dict; // TO STORE RESIDUAL AND INTERPOLANT AT EACH LEVEL
};

void solve(ProblemVar& obj);

#endif