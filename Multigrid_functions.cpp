/*
TODO : Pybind
*/
#include <vector>
#include "oneapi/mkl.hpp"
#include "mkl_types.h"
#include <CL/sycl.hpp>
#include <Eigen/SparseLU>
#include <unordered_map>
#include <cmath>
#include <tuple>

using namespace sycl;
using namespace oneapi::mkl::sparse;

class ProblemVar {
//Object of this class is specific to a problem and is initialized for each problem from Python interface.
public:
	Eigen::SparseMatrix<double> coarsest_level_matrix;
	std::unordered_map<int, csr_matrix_elements> A_sp_dict;
	std::unordered_map<int, csr_jacobi_elements> A_jacobi_sp_dict;
	std::unordered_map<int, std::unordered_map<int, int>> topo_to_space_dict;
	std::unordered_map<int, std::vector<double>> b_dict;
	std::unordered_map<int, std::vector<std::tuple<int, int>>>parent_info_dict;
	std::unordered_map<int, std::unordered_map<int, std::vector<int>>>coarse_grid_edges_dict;	
};

struct csr_jacobi_elements 
{
	csr_matrix_elements D_inv;
	csr_matrix_elements R_omega;
};

struct csr_matrix_elements
{
	std::vector<int> row;
	std::vector<int>col;
	std::vector<double> values;
	matrix_handle_t matrix_handle;
	int size;
};

int coarsest_level = 0;
int finest_level = 5;
int mu0 = 2;
int mu1 = 1;
int mu2 = 1;
float omega = 4 / 5;
					
Eigen::SparseMatrix<double> coarse_matrix_assemble(std::vector<int>& cols, std::vector<int>& rows, 
	std::vector<double>& vals, int &num_row) {
	Eigen::SparseMatrix<double> sparse_mat(num_row, num_row);
	sparse_mat.reserve(Eigen::VectorXi::Constant(num_row, 4));
	for (int i = 0; i < rows.size()-1; i++) {

		for (int j = rows[i]; j <= rows[i+1]-1; j++) {
			sparse_mat.insert(i, cols[j]) = vals[j];
		}
	}
	return sparse_mat;
}

std::vector<double> direct_solver(Eigen::SparseMatrix<double>& sparse_matrix, std::vector<double>& b) {
	Eigen::VectorXd b_eg = Eigen::VectorXd::Map(b.data(), b.size());
	Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
	solver.compute(sparse_matrix);
	Eigen::VectorXd solution = solver.solve(b_eg);
	std::vector<double> sol;
	sol.resize(solution.size());
	Eigen::VectorXd::Map(&sol[0], solution.size()) = solution;
	return sol;
}


void jacobirelaxation(cl::sycl::queue &q, std::vector<double>& vec, std::vector<double> &b, ProblemVar &obj 
	, int current_level) {
	int size = vec.size();
	std::vector<double> ans(size, 0.0);
	std::vector<double> prod_1(size, 0.0);
	std::vector<double> prod_2(size, 0.0);

	cl::sycl::event R_V_mul_done = cl::sycl::event();
	cl::sycl::event D_inv_F_done = cl::sycl::event(); // Replace it by a vector to vector multiplication
	cl::sycl::event add_done = cl::sycl::event();

	for (int i = 0; i < 10; i++) {
		R_V_mul_done = gemv(q, oneapi::mkl::transpose::nontrans, 1.0, obj.A_jacobi_sp_dict[current_level].R_omega.matrix_handle,
			vec.data(), 0.0, prod_1.data());
		D_inv_F_done = gemv(q, oneapi::mkl::transpose::nontrans, omega, obj.A_jacobi_sp_dict[current_level].D_inv.matrix_handle,
			b.data(), 0.0, prod_2.data());
		add_done = oneapi::mkl::vm::add(q, size, prod_1.data(), prod_2.data(), vec.data(),
			{ R_V_mul_done , D_inv_F_done });
		add_done.wait();
	}
	//Check if the vec gets assigned back to the same in the last statement or not!
}

std::vector<double> interpolation2D(std::vector<double>vec_2h, ProblemVar &obj , int target_level) {
	int vec_2h_dim = vec_2h.size();
	int vec_h_dim = int(std::pow(2 * (std::sqrt(vec_2h_dim) - 1) + 1, 2));
	std::vector<double> vec_h(vec_h_dim);
	for (int i = 0; i < vec_h_dim; i++) {
		int fine_space_dof = obj.topo_to_space_dict[target_level][i];
		if (std::get<0>(obj.parent_info_dict[target_level][i]) == 0) {
			// Coarse topo dof and fine topo dof coincide
			vec_h[fine_space_dof] = vec_2h[obj.topo_to_space_dict[target_level-1][std::get<1>(obj.parent_info_dict[target_level][i])]];
		}
		else {
			//Fine dof lies on coarse edge
			int edge_num = std::get<1>(obj.parent_info_dict[target_level][i]);
			//Obtain corresponding edge vertices on topology
			int coarse_dof_1 = obj.topo_to_space_dict[target_level - 1][obj.coarse_grid_edges_dict[target_level-1][edge_num][0]];
			int coarse_dof_2 = obj.topo_to_space_dict[target_level - 1][obj.coarse_grid_edges_dict[target_level-1][edge_num][1]];
			vec_h[fine_space_dof] = 0.5 * (vec_2h[coarse_dof_1] + vec_2h[coarse_dof_2]);
		}
	}
	return vec_h;
}

//Define the restriction operator

std::vector<double> restriction2D(std::vector<double>vec_h , ProblemVar &obj , int target_level) {
	int vec_h_dim = vec_h.size();
	int vec_2h_dim = int(std::pow(((std::sqrt(vec_h_dim) - 1) / 2) + 1, 2));
	std::vector<double> vec_2h(vec_2h_dim);
	for (int i = 0; i < vec_2h_dim; i++) {
		vec_2h[obj.topo_to_space_dict[target_level][i]] = vec_h[obj.topo_to_space_dict[target_level+1][i]];
	}
	return vec_2h;
}

std::vector<double> vcyclemultigrid(cl::sycl::queue& q, ProblemVar &obj, 
	std::vector<double>& vec_h, std::vector<double>& f_h , int current_level) {

	std::int32_t vec_size = vec_h.size();
	std::vector<double> vec_2h;
	if (current_level == coarsest_level) {		//USE DIRECT SOLVER HERE
		vec_h = direct_solver(obj.coarsest_level_matrix, f_h);
		return vec_h;
	}
	else {
		///////Perform one smoothing operation here first
		jacobirelaxation(q, vec_h, f_h, obj, current_level);

		std::vector<double> temp_vec1(vec_size, 0);
		std::vector<double> residual(vec_size, 0);
		std::vector<double> f_2h;
		cl::sycl::event gemv_A_vec_done = cl::sycl::event();
		cl::sycl::event sub_done = cl::sycl::event();
		gemv_A_vec_done = gemv(q, oneapi::mkl::transpose::nontrans, 1.0, obj.A_sp_dict[current_level].matrix_handle,
			vec_h.data(),0.0, temp_vec1.data(), {});
		sub_done = oneapi::mkl::vm::sub(q, vec_size, f_h.data(), temp_vec1.data(), residual.data(), { gemv_A_vec_done });
		sub_done.wait();

		//applying the restriction on the residual
		f_2h = restriction2D(residual , obj , current_level-1);
		std::int32_t f_2h_size = f_2h.size();
		vec_2h = std::vector<double>(f_2h_size, 0);
		vec_2h = vcyclemultigrid(q, obj, vec_2h, f_2h , current_level-1);
	}
	std::vector<double> vec_2h_interpolation = interpolation2D(vec_2h, obj,current_level);
	cl::sycl::event vec_h_addition = cl::sycl::event();
	std::vector<double> vec_h_final(vec_size, 0);
	vec_h_addition = oneapi::mkl::vm::add(q, vec_size, vec_h.data(), vec_2h_interpolation.data(), 
		vec_h_final.data(), {});
	// Check if you can add vec = vec + vec_2h_interpolation here
	vec_h_addition.wait();

	// Perform jacobi relaxation here 
	//////////////////Check for the vec_h_final or vec_h directly
	jacobirelaxation(q, vec_h_final, f_h, obj, current_level);
	return vec_h_final;
}

std::vector<double> fullmultigrid(cl::sycl::queue& q, ProblemVar &object, std::vector<double>& f_h , int current_level) {
	std::vector<double> vec_h(f_h.size(), 0);
	std::vector<double> vec_2h;
	if (current_level == coarsest_level) {
		// Use direct solver here
		vec_h = direct_solver(object.coarsest_level_matrix, f_h);
		return vec_h;
	}
	else {
		vec_2h = fullmultigrid(q, object, object.b_dict[current_level-1] , current_level-1);
	}
	vec_h = interpolation2D(vec_2h , object , current_level);
	for (std::int32_t i = 0; i <= mu0; i++) {
		vec_h = vcyclemultigrid(q,object,vec_h,f_h,current_level);
	}
	return vec_h;
}

std::vector<double> multigrid_solver(ProblemVar& object) {
	cl::sycl::queue q;
	std::vector<double> answer = fullmultigrid(q, object , object.b_dict[finest_level], finest_level );
	return answer;
}