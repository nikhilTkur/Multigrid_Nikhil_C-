#include <CL/sycl.hpp>
#include "oneapi/mkl.hpp"
#include "mkl_types.h"
#include <Eigen/SparseLU>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>

using namespace sycl;
using namespace oneapi::mkl::sparse;
namespace py = pybind11;

int coarsest_level = 1;
int finest_level = 5;
int mu0 = 2;
int mu1 = 1;
int mu2 = 1;
float omega = 4 / 5;

struct csr_jacobi_elements
{
	csr_matrix_elements D_inv;
	csr_matrix_elements R_omega;
	csr_jacobi_elements() {}
	csr_jacobi_elements(csr_matrix_elements&D_inv_ , csr_matrix_elements &R_omega_){
		D_inv = D_inv_;
		R_omega = R_omega_;
	}
};

struct csr_matrix_elements
{
	std::vector<int> row;
	std::vector<int>col;
	std::vector<double> values;
	matrix_handle_t matrix_handle;
	int size;
	csr_matrix_elements() {}
	csr_matrix_elements(py::array_t<int>& row_, py::array_t<int>& col_, py::array_t<double>& vals_, int& size_) {
		row = std::vector<int> (row_.data(), row_.data() + row_.size());
		col = std::vector<int> (col_.data(), col_.data() + col_.size());
		values = std::vector<double> (vals_.data(), vals_.data() + vals_.size());
		// Create a matrix entity
		size = size_;
		init_matrix_handle(&matrix_handle);
		set_csr_data(matrix_handle, size, size, oneapi::mkl::index_base::zero, row.data(),
			col.data(), values.data());
	}
};

class ProblemVar {
//Object of this class is specific to a problem and is initialized for each problem from Python interface.
public:
	Eigen::SparseMatrix<double> coarsest_level_matrix;
	std::vector<csr_matrix_elements> A_sp_dict;
	std::vector<csr_jacobi_elements> A_jacobi_sp_dict;

	//[level -> [index = topo dof , value = space dof]]
	std::unordered_map<int, std::vector<int>> topo_to_space_dict;
	std::unordered_map<int, std::vector<double>> b_dict;

	//[level -> [index = fine dof , value = matching coarse dof]]
	std::unordered_map<int, std::vector<int>>parent_info_vertex_dict;

	//[level -> [index = fine dof - vec_2h_dim , value = matching coarse edge]]
	std::unordered_map<int, std::vector<int>>parent_info_edges_dict;

	//[level -> [2 * index = coarse_grid edge , value = coarse_dofs]] eg [1 -> [1,2,3,4,5,6]] -> edge 0 = 1,2 | edge 1 = 3,4, also, edges index from 0 to no_of_edges
	std::unordered_map<int, std::vector<int>>coarse_grid_edges_dict;

	//Defining a constructor that takes in py::arrays and assigns it to above member variables
	ProblemVar(Eigen::SparseMatrix<double>& coarsest_level_matrix_py, py::array_t<csr_matrix_elements>& A_sp_list_py,
		py::array_t<csr_jacobi_elements>& A_jacobi_sp_list_py,
		py::array_t<py::array_t<int>>& topo_to_space_list_py,
		py::array_t<py::array_t<double>>& b_list_py,
		py::array_t<py::array_t<int>>& parent_info_vertex_list_py,
		py::array_t<py::array_t<int>>& parent_info_edges_list_py,
		py::array_t<py::array_t<int>>& coarse_grid_edges_list_py) {

		coarsest_level_matrix = coarsest_level_matrix_py;
		//assigning dicts of matrices
		A_sp_dict = std::vector<csr_matrix_elements>(A_sp_list_py.data(), A_sp_list_py.data() + A_sp_list_py.size());
		A_jacobi_sp_dict = std::vector<csr_jacobi_elements>(A_jacobi_sp_list_py.data() , A_jacobi_sp_list_py.data() +
			A_jacobi_sp_list_py.size());
		//initializing topo_to_space_dict

		auto r1 = topo_to_space_list_py.mutable_unchecked(); // can now index over first py::array_t
		auto r2 = b_list_py.mutable_unchecked();
		auto r3 = parent_info_vertex_list_py.mutable_unchecked();
		auto r4 = parent_info_edges_list_py.mutable_unchecked();
		auto r5 = coarse_grid_edges_list_py.mutable_unchecked();
		for (int i = coarsest_level; i <= finest_level; i++) {
			//Copy the topo_list for a level
			topo_to_space_dict[i] = std::vector<int>(r1(i).data() , r1(i).data()+r1(i).size());
			b_dict[i] = std::vector<double>(r2(i).data(), r2(i).data() + r2(i).size());
			if (i != coarsest_level) {
				parent_info_vertex_dict[i+1] = std::vector<int>(r3(i).data(), r3(i).data() + r3(i).size());
				parent_info_edges_dict[i+1] = std::vector<int>(r4(i).data(), r4(i).data() + r4(i).size());
			}
			if (i != finest_level) {
				coarse_grid_edges_dict[i] = std::vector<int>(r5(i).data(), r5(i).data() + r5(i).size());
			}
		}
	}
};
					
Eigen::SparseMatrix<double> coarse_matrix_assemble(py::array_t<int>& rows_, py::array_t<int>& cols_,
	py::array_t<double>& vals_, int &num_row) {
	std::vector<int> cols = std::vector<int>(cols_.data(), cols_.data() + cols_.size());
	std::vector<int> rows = std::vector<int>(rows_.data(), rows_.data() + rows_.size());
	std::vector<double> vals = std::vector<double>(vals_.data(), vals_.data() + vals_.size());
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
	//std::vector<double> ans(size, 0.0);
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
}

std::vector<double> interpolation2D(std::vector<double>vec_2h, ProblemVar &obj , int target_level) {
	int vec_2h_dim = vec_2h.size();
	int vec_h_dim = int(std::pow(2 * (std::sqrt(vec_2h_dim) - 1) + 1, 2));
	std::vector<double> vec_h(vec_h_dim);
	for (int i = 0; i < vec_2h_dim; i++) {
		int fine_space_dof = obj.topo_to_space_dict[target_level][i];
		// Coarse topo dof and fine topo dof coincide. As per the definition of refinement, coarse topo dof coinciding with
		// fine topo dof have the same number in topo. 
		vec_h[fine_space_dof] = vec_2h[obj.topo_to_space_dict[target_level-1][(obj.parent_info_vertex_dict[target_level][i])]];
	}
	for (int i = vec_2h_dim ; i < vec_h_dim ; i++){
		int fine_space_dof = obj.topo_to_space_dict[target_level][i];
		//Fine dof lies on a coarse edge
		// Parent info stores the edges in increasing order
		int edge_num = (obj.parent_info_edges_dict[target_level][i]); 
		//Obtain corresponding edge vertices on topology and converting it to space dof
		int coarse_dof_1 = obj.topo_to_space_dict[target_level - 1][obj.coarse_grid_edges_dict[target_level-1][2*edge_num]];
		int coarse_dof_2 = obj.topo_to_space_dict[target_level - 1][obj.coarse_grid_edges_dict[target_level-1][2*edge_num+1]];
		vec_h[fine_space_dof] = 0.5 * (vec_2h[coarse_dof_1] + vec_2h[coarse_dof_2]);
	}
	return vec_h;
}

std::vector<double> restriction2D(std::vector<double>vec_h, ProblemVar& obj, int target_level) {
	int vec_h_dim = vec_h.size();
	int vec_2h_dim = int(std::pow(((std::sqrt(vec_h_dim) - 1) / 2) + 1, 2));
	std::vector<double> vec_2h(vec_2h_dim);
	for (int i = 0; i < vec_2h_dim; i++) {
		vec_2h[obj.topo_to_space_dict[target_level][i]] = vec_h[obj.topo_to_space_dict[target_level + 1][i]];
	}
	return vec_2h;
}

std::vector<double> interpolation2D_parallel(cl::sycl::queue &q , std::vector<double>vec_2h, ProblemVar& obj, int target_level) {
	size_t vec_2h_dim = vec_2h.size();
	size_t vec_h_dim = size_t(std::pow(2 * (std::sqrt(vec_2h_dim) - 1) + 1, 2));
	std::vector<double> vec_h(vec_h_dim);
	{
		buffer t_to_s_fine{ obj.topo_to_space_dict[target_level] };
		buffer t_to_s_coarse{ obj.topo_to_space_dict[target_level - 1] };
		buffer parent_info_vertex{ obj.parent_info_vertex_dict[target_level] };
		buffer parent_info_edges{ obj.parent_info_edges_dict[target_level] };
		buffer coarse_grid_edges{ obj.coarse_grid_edges_dict[target_level - 1] };
		buffer vec_h_buf{ vec_h };
		buffer vec_2h_buf{ vec_2h };

		q.submit([&](handler& h) {
			accessor vec_2h_acc{ vec_2h_buf , h };
			accessor vec_h_acc{ vec_h_buf,h };
			accessor t_s_fine{ t_to_s_fine,h };
			accessor t_s_coarse{ t_to_s_coarse,h };
			accessor parent_info_vertex_acc{ parent_info_vertex,h };
			accessor parent_info_edges_acc{ parent_info_edges,h };
			accessor coarse_edges{ coarse_grid_edges,h };

			h.parallel_for(range<1>{vec_2h_dim}, [=](id<1>idx) {
				size_t fine_space_dof = t_s_fine[idx[0]];
				vec_h_acc[fine_space_dof] = vec_2h_acc[t_s_coarse[(parent_info_vertex_acc[idx[0]])]];
			});

			h.parallel_for(range<1>{vec_h_dim - vec_2h_dim}, [=](id<1>idx) {
				size_t fine_topo_dof = idx[0] + vec_2h_dim;
				size_t fine_space_dof = t_s_fine[fine_topo_dof];
				size_t edge_num = parent_info_edges_acc[fine_topo_dof];
				size_t coarse_dof_1 = t_s_coarse[coarse_edges[2*edge_num]];
				size_t coarse_dof_2 = t_s_coarse[coarse_edges[2*edge_num+1]];
				vec_h_acc[fine_space_dof] = 0.5 * (vec_2h_acc[coarse_dof_1] + vec_2h_acc[coarse_dof_2]);				
			});				
		});
	}
	return vec_h;
}

//Define the restriction operator

std::vector<double> restriction2D_parallel(cl::sycl::queue& q, std::vector<double>vec_h , ProblemVar &obj , int target_level) {
	int vec_h_dim = vec_h.size();
	int vec_2h_dim = int(std::pow(((std::sqrt(vec_h_dim) - 1) / 2) + 1, 2));
	std::vector<double> vec_2h(vec_2h_dim);
	{
		buffer t_to_s_fine{ obj.topo_to_space_dict[target_level + 1] };
		buffer t_to_s_coarse{ obj.topo_to_space_dict[target_level] };
		buffer vec_h_buf{ vec_h };
		buffer vec_2h_buf{ vec_2h };

		q.submit([&](handler& h) {
			accessor vec_2h_acc{ vec_2h_buf , h };		
			accessor vec_h_acc{ vec_h_buf,h };
			accessor t_s_fine{ t_to_s_fine,h };
			accessor t_s_coarse{ t_to_s_coarse,h };

			h.parallel_for(range <1>{vec_2h_dim}, [=](id<1>idx) {
				vec_2h_acc[t_s_coarse[idx[0]]] = vec_h_acc[t_to_s_fine[idx[0]]];
			});
		});
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
		//Perform one smoothing operation here first
		jacobirelaxation(q, vec_h, f_h, obj, current_level);
		//std::vector<double> temp_vec1(vec_size, 0);
		std::vector<double> residual(vec_size, 0);
		std::vector<double> f_2h;
		cl::sycl::event gemv_A_vec_done = cl::sycl::event();
		cl::sycl::event sub_done = cl::sycl::event();
		gemv_A_vec_done = gemv(q, oneapi::mkl::transpose::nontrans, 1.0, obj.A_sp_dict[current_level].matrix_handle,
			vec_h.data(),0.0, residual.data(), {});
		sub_done = oneapi::mkl::vm::sub(q, vec_size, f_h.data(), residual.data(), residual.data(), { gemv_A_vec_done });
		sub_done.wait();

		//applying the restriction on the residual
		//CASE-WISE Restriction to be used for parallel version - TODO
		f_2h = restriction2D(residual , obj , current_level-1);

		//Perform element wise multiplication with 4 / 8 depending upon the dimention of problem and to account for basis fns.
		std::int32_t f_2h_size = f_2h.size();
		vec_2h = std::vector<double>(f_2h_size, 0);
		vec_2h = vcyclemultigrid(q, obj, vec_2h, f_2h , current_level-1);
	}

	//CASE-WISE Interpolation to be used for parallel version
	std::vector<double> vec_2h_interpolation = interpolation2D(vec_2h, obj,current_level);
	cl::sycl::event vec_h_addition = cl::sycl::event();
	//std::vector<double> vec_h_final(vec_size, 0);
	vec_h_addition = oneapi::mkl::vm::add(q, vec_size, vec_h.data(), vec_2h_interpolation.data(), 
		vec_h.data(), {});
	vec_h_addition.wait();

	// Perform jacobi relaxation here
	jacobirelaxation(q, vec_h, f_h, obj, current_level);
	return vec_h;
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

std::vector<double> solve(ProblemVar& object) {
	cl::sycl::queue q;
	std::vector<double> answer = fullmultigrid(q, object , object.b_dict[finest_level], finest_level );
	return answer;
}

PYBIND11_MODULE(multigrid_solver, m) {
	py::class_<csr_matrix_elements>(m, "csr_matrix_elememts")
		.def(py::init<py::array_t<int>&, py::array_t<int>&, py::array_t<double>&, int&>());
	py::class_<csr_jacobi_elements>(m, "csr_jacobi_elements")
		.def(py::init<csr_matrix_elements&, csr_matrix_elements&>());
	py::class_<ProblemVar>(m, "ProblemVar")
		.def(py::init<Eigen::SparseMatrix<double>&,
			py::array_t<csr_matrix_elements>&,
			py::array_t<csr_jacobi_elements>&,
			py::array_t<py::array_t<int>>&,
			py::array_t<py::array_t<double>>&,
			py::array_t<py::array_t<int>>&,
			py::array_t<py::array_t<int>>&,
			py::array_t<py::array_t<int>>&>());
	m.def("eigen_matrix_assemble", &coarse_matrix_assemble);
	m.def("solve", &solve);

}