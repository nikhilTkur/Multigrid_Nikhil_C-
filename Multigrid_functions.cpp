#include <CL/sycl.hpp>
#include "oneapi/mkl.hpp"
#include "mkl_types.h"
#include <Eigen/SparseLU>
#include <vector>
#include <unordered_map>

int coarsest_level = 1;
int finest_level = 5;
int mu0 = 5; // V-cycles per  finest_level
int mu0_1 = 1; // V-cycles per other levels
int mu1 = 3; // pre and post smooths
double omega = 4 / 5; // Jacobi smoother parameter

struct csr_matrix_elements
{
	cl::sycl::buffer<std::int32_t, 1> row;
	cl::sycl::buffer<std::int32_t, 1> col;
	cl::sycl::buffer<double, 1> values;
	oneapi::mkl::sparse::matrix_handle_t matrix;
	std::int32_t size;
	csr_matrix_elements(std::int32_t *row_ptr , std::int32_t *col_ptr , double *val_ptr , std::int32_t nnz , std::int32_t n_dofs) {
		row = cl::sycl::buffer<std::int32_t, 1>{ row_ptr , cl::sycl::range<1>{n_dofs + 1} };
		col = cl::sycl::buffer<std::int32_t, 1>{ col_ptr , cl::sycl::range<1>{nnz} };
		values = cl::sycl::buffer<double, 1>{ val_ptr, cl::sycl::range<1>{nnz} };
		// Create a matrix entity
		size = n_dofs;
		oneapi::mkl::sparse::init_matrix_handle(&matrix);
		oneapi::mkl::sparse::set_csr_data(matrix, size, size, oneapi::mkl::index_base::zero, row,col, values);
	}
};

class ProblemVar {
public:
	Eigen::SparseMatrix<double> coarsest_level_matrix;
	std::vector<csr_matrix_elements> A_sp_dict;

	//[level -> [index = topo dof , value = space dof]]
	std::unordered_map<int, cl::sycl::buffer<std::int32_t , 1>> topo_to_space_dict;

	// [level -> RHS for that level as a buffer object]
	std::unordered_map<int, cl::sycl::buffer<double, 1>> b_dict;

	std::unordered_map<int, std::int32_t> num_dofs_per_level;

	//[level -> [index = fine dof , value = matching coarse dof from parent grid]]
	std::unordered_map<int, cl::sycl::buffer<std::int32_t, 1>>parent_info_vertex_dict;

	//[level -> [index = fine dof - vec_2h_dim , value = matching coarse edge from parent grid]]
	std::unordered_map<int, cl::sycl::buffer<std::int32_t, 1>>parent_info_edges_dict;

	//[level -> [2 * index = coarse_grid edge , value = coarse_dofs]] eg [1 -> [1,2,3,4,5,6]] -> edge 0 = 1,2 | edge 1 = 3,4, also, edges index from 0 to no_of_edges
	std::unordered_map<int, cl::sycl::buffer<std::int32_t, 1>>coarse_grid_edges_dict;

	std::unordered_map<int, cl::sycl::buffer<double, 1>> vecs_dict;
	std::unordered_map<int, cl::sycl::buffer<double, 1>> temp_dict; // TO STORE RESIDUAL AND INTERPOLANT AT EACH LEVEL
};
void coarse_matrix_assemble(cl::sycl::queue &q,ProblemVar &obj) {

	// Assembles the coarsest level matrix as an Eigen matrix. Coarse domain to be solved sequentially

	q.submit([&](cl::sycl::handler &h) {
		//Accessors for coarsest level CSR data.
		auto row = obj.A_sp_dict[coarsest_level].row.get_access<cl::sycl::access::mode::read>(h);
		auto col = obj.A_sp_dict[coarsest_level].col.get_access<cl::sycl::access::mode::read>(h);
		auto val = obj.A_sp_dict[coarsest_level].values.get_access<cl::sycl::access::mode::read>(h);

		obj.coarsest_level_matrix = Eigen::SparseMatrix<double>(obj.num_dofs_per_level[coarsest_level], 
			obj.num_dofs_per_level[coarsest_level]);
		obj.coarsest_level_matrix.reserve(Eigen::VectorXd::Constant(obj.num_dofs_per_level[coarsest_level], 4));

		for (int i = 0; i < obj.num_dofs_per_level[coarsest_level]; i++) {
			for (int j = row[i]; j <= row[i+1]-1; j++) {
				obj.coarsest_level_matrix.insert(i, col[j]) = val[j];
			}
		}
	});
}
void direct_solver(cl::sycl::queue &q , ProblemVar &obj) {
	// direct solver for the coarsest level grid.
	q.submit([&](cl::sycl::handler &h) {
		auto coarse_vec = obj.vecs_dict[coarsest_level].get_access<cl::sycl::access::mode::write>(h); // Stores the solution vector
		auto coarse_b = obj.b_dict[coarsest_level].get_access<cl::sycl::access::mode::read>(h); // stores the RHS of Ax = b
		Eigen::VectorXd b_eigen = Eigen::VectorXd::Map(coarse_b.get_pointer());
		Eigen::VectorXd vec_eigen = Eigen::VectorXd::Map(coarse_vec.get_pointer());
		Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
		solver.compute(obj.coarsest_level_matrix);
		vec_eigen = solver.solve(b_eigen);
	});
}

void jacobi_relaxation(cl::sycl::queue &q,ProblemVar& obj, int current_level) {
	// Iterates on the current level on obj.vecs_dict[current_level] buffer. Stores intermediate value in temp_dict[current_level]
	// and then adds the result back to the obj.vecs_dict[current_level].

	for (int iterations = 1; iterations <= mu1; iterations++) {
		// TODO	  =>		v(k+1) = [(1 - omega) x I + omega x D^-1 x(-L-U)] x v(k) + omega x D^-1 x f
		//					
		// step 1 =>		v* = (-L-U) x v
		// step 2 =>		v* = D^-1 x (v* + f)
		// step 3 =>		v = (1-omega) x v + omega x v*

		q.submit([&](cl::sycl::handler& h) {
			// Accessor for current_level matrix CSR values
			auto row = obj.A_sp_dict[current_level].row.get_access<cl::sycl::access::mode::read>(h);
			auto col = obj.A_sp_dict[current_level].col.get_access<cl::sycl::access::mode::read>(h);
			auto val = obj.A_sp_dict[current_level].values.get_access<cl::sycl::access::mode::read>(h);

			// Accessor for current_level vector
			auto vec = obj.vecs_dict[current_level].get_access<cl::sycl::access::mode::read>(h);
			auto vec_star = obj.temp_dict[current_level].get_access<cl::sycl::access::mode::read_write>(h);
			auto f = obj.b_dict[current_level].get_access<cl::sycl::access::mode::read>(h);

			h.parallel_for(cl::sycl::range<1>{obj.num_dofs_per_level[current_level]}, [=](cl::sycl::id<1>idx) {
				// parallely launch all the dofs on device.
				//current dof = idx[0]
				double diag_multiplier = 0;
				for (std::int32_t i = row[idx[0]]; i < row[idx[0] + 1]; i++) {
					if (col[i] != idx[0]) { // means that its a non-diagonal entry in matrix => step 1 
						vec_star[idx[0]] += -1.0 * val[i] * vec[col[i]];
					}
					else {
						diag_multiplier = 1.0 / val[i];
					}
				}
				vec_star[idx[0]] = diag_multiplier * (vec_star[idx[0]] + f[idx[0]]);	// step 2
			});

		});
		q.wait();
		q.submit([&](cl::sycl::handler& h) {
			// Accessor for current_level vector
			auto vec = obj.vecs_dict[current_level].get_access<cl::sycl::access::mode::read_write>(h);
			auto vec_star = obj.temp_dict[current_level].get_access<cl::sycl::access::mode::read>(h);

			h.parallel_for(cl::sycl::range<1>{obj.num_dofs_per_level[current_level]}, [=](cl::sycl::id<1>idx) {

				vec[idx[0]] = (1.0 - omega) * vec[idx[0]] + omega * vec_star[idx[0]]; //	step 3
			});
		});
		q.wait();
	}
}

std::vector<double> interpolation2D(cl::sycl::queue &q, ProblemVar &obj , int target_level) {
	// interpolates the values from vecs_dict[target_level -1] -> temp_dict[target_level]

	q.submit([&](cl::sycl::handler &h) {
		// Accessor for temp_dict[target_level] ->WIRTE-ONLY
		auto vec_h = obj.temp_dict[target_level].get_access<cl::sycl::access::mode::write>(h);

		// Accessor for vecs_dict[target_level -1] ->READ-ONLY
		auto vec_2h = obj.vecs_dict[target_level - 1].get_access<cl::sycl::access::mode::read>(h);

		//Accessor for topo_to_space_dict_coarse ->READ-ONLY
		auto t_2_s_coarse = obj.topo_to_space_dict[target_level - 1].get_access<cl::sycl::access::mode::read>(h);

		//Accessor for topo_to_space_dict_fine ->READ-ONLY
		auto t_2_s_fine = obj.topo_to_space_dict[target_level].get_access<cl::sycl::access::mode::read>(h);

		//Accessor for parent_info_dicts for current level ->READ-ONLY
		auto parent_info_vertices = obj.parent_info_vertex_dict[target_level].get_access<cl::sycl::access::mode::read>(h);
		auto parent_info_edges = obj.parent_info_edges_dict[target_level].get_access<cl::sycl::access::mode::read>(h);

		//Accessor for coarse_edges ->READ-ONLY
		auto coarse_level_edges = obj.coarse_grid_edges_dict[target_level - 1].get_access<cl::sycl::access::mode::read>(h);

		if (obj.num_dofs_per_level[target_level] > 1000000) { //Use parallel version of interpolation

			//directly inject the value from coarse grid dof to fine grid dof as they coincide
			h.parallel_for(cl::sycl::range<1>{obj.num_dofs_per_level[target_level-1]}, [=](cl::sycl::id<1>idx) {				
				//find the fine space dof from topology map
				std::int32_t fine_space_dof = t_2_s_fine[idx[0]];
				vec_h[fine_space_dof] = vec_2h[t_2_s_coarse[(parent_info_vertices[idx[0]])]];
			});
			// interpolate the value for fine dofs on the edges of coarse dofs
			h.parallel_for(cl::sycl::range<1>{obj.num_dofs_per_level[target_level] - obj.num_dofs_per_level[target_level-1]},
			[=](cl::sycl::id<1>idx) {
				std::int32_t fine_topo_dof = idx[0] + obj.num_dofs_per_level.at(target_level-1);
				std::int32_t fine_space_dof = t_2_s_fine[fine_topo_dof];
				std::int32_t edge_num = parent_info_edges[fine_topo_dof];
				std::int32_t coarse_dof_1 = t_2_s_coarse[coarse_level_edges[2 * edge_num]];
				std::int32_t coarse_dof_2 = t_2_s_coarse[coarse_level_edges[2 * edge_num + 1]];
				vec_h[fine_space_dof] = 0.5 * (vec_2h[coarse_dof_1] + vec_2h[coarse_dof_2]);
			});
		}
		else { // use the serial version of the interpolation

			for (std::int32_t i = 0; i < obj.num_dofs_per_level[target_level - 1]; i++) {
				std::int32_t fine_space_dof = t_2_s_fine[i];
				vec_h[fine_space_dof] = vec_2h[t_2_s_coarse[(parent_info_vertices[i])]];
			}
			for (std::int32_t i = obj.num_dofs_per_level[target_level-1]; i < obj.num_dofs_per_level[target_level]; i++) {
				std::int32_t fine_space_dof = t_2_s_fine[i];
				std::int32_t edge_num = parent_info_edges[i];
				//Obtain corresponding edge vertices on topology and converting it to space dof
				std::int32_t coarse_dof_1 = t_2_s_coarse[coarse_level_edges[2 * edge_num]];
				std::int32_t coarse_dof_2 = t_2_s_coarse[coarse_level_edges[2 * edge_num + 1]];
				vec_h[fine_space_dof] = 0.5 * (vec_2h[coarse_dof_1] + vec_2h[coarse_dof_2]);
			}
		}		
	});
	q.wait();	
}

void restriction2D(cl::sycl::queue &q, ProblemVar &obj, int target_level) {

	q.submit([&](cl::sycl::handler& h) {
		// Accessor for temp_dict[target_level + 1] ->READ-ONLY
		auto vec_h = obj.temp_dict[target_level+1].get_access<cl::sycl::access::mode::read>(h);

		// Accessor for b_dict[target_level] ->WRITE-ONLY
		auto vec_2h = obj.b_dict[target_level].get_access<cl::sycl::access::mode::write>(h);

		//Accessor for topo_to_space_dict_coarse ->READ-ONLY
		auto t_2_s_coarse = obj.topo_to_space_dict[target_level].get_access<cl::sycl::access::mode::read>(h);

		//Accessor for topo_to_space_dict_fine ->READ-ONLY
		auto t_2_s_fine = obj.topo_to_space_dict[target_level+1].get_access<cl::sycl::access::mode::read>(h);

		if(obj.num_dofs_per_level[target_level] > 1000000) { // use parallel restriction
			h.parallel_for(cl::sycl::range <1>{obj.num_dofs_per_level[target_level]}, [=](cl::sycl::id<1>idx) {
				vec_2h[t_2_s_coarse[idx[0]]] = vec_h[t_2_s_fine[idx[0]]];
			});
		}
		else { // use serial restriction
			for (std::int32_t i = 0; i < obj.num_dofs_per_level[target_level]; i++) {
				vec_2h[t_2_s_coarse[i]] = vec_h[t_2_s_fine[i]];
			}
		}
	});
	q.wait();
}

void vcyclemultigrid(cl::sycl::queue& q, ProblemVar &obj, int current_level) {

	if (current_level == coarsest_level) {		
		//Use direct solver here
		direct_solver(q , obj);
	}
	else {
		//Perform one smoothing operation here first
		jacobi_relaxation(q, obj, current_level);
		oneapi::mkl::sparse::gemv(q, oneapi::mkl::transpose::nontrans, 1.0, obj.A_sp_dict[current_level].matrix,
			obj.vecs_dict[current_level],0.0, obj.temp_dict[current_level]);
		oneapi::mkl::vm::sub(q, obj.num_dofs_per_level[current_level], obj.b_dict[current_level], obj.temp_dict[current_level]
			, obj.temp_dict[current_level]);

		restriction2D(q,obj,current_level-1);
		// make vecs_2h '0' so that it stores the current best approximation of the error
		q.submit([&](cl::sycl::handler& h) {
			auto vec_2h = obj.vecs_dict[current_level-1].get_access<cl::sycl::access::mode::write>(h);
			h.parallel_for(cl::sycl::range<1>{obj.num_dofs_per_level[current_level-1]}, [=](cl::sycl::id<1>idx) {
				vec_2h[idx[0]] = 0.0;
			});
		});
		q.wait();
		
		//Perform element wise multiplication with 4 / 8 depending upon the dimention of problem and to account for basis fns.
		vcyclemultigrid(q, obj, current_level-1);
	}
	interpolation2D(q,obj,current_level);
	oneapi::mkl::vm::add(q, obj.num_dofs_per_level[current_level], obj.vecs_dict[current_level], obj.temp_dict[current_level],
		obj.vecs_dict[current_level]);
	// Perform jacobi relaxation here
	jacobi_relaxation(q, obj, current_level);
}

void fullmultigrid(cl::sycl::queue& q, ProblemVar &obj, int current_level) {
/*
						4-LEVEL-FMG CYCLE WITH V-CYCLE
  Levels

	1	 \																	   / V=>S\								  /=>S=>FINAL SOLUTION
																			  I		  R								 I
	2	   \								 / V=>S\					 /=>S/		   \=>S\					/=>S/
											I		R					I					R				   I
	3		 \		   / V=>S\		  / =>S/		 \=>S\		  / =>S/					 \=>S\		  /=>S/
					  I		  R	     I					  R		 I								  R		 I
	4		   \_ DS_/		   \_DS_/					   \_DS_/								   \_DS_/
	
	DS => DIRECT SOLVER (EIGEN SPARSELU)
	I => INTERPOLATION
	R => RESTRICTION
	S => SMOOTHER (JACOBI)	
*/

	if (current_level == coarsest_level) {
		// Use direct solver here
		direct_solver(q , obj);
	}
	else {
		fullmultigrid(q, obj, current_level-1);
	}
	interpolation2D(q,obj , current_level);
	//Transfer temp_vec to vecs so that vecs acts as a good initial guess based on previous grid soln.
	q.submit([&](cl::sycl::handler& h) {
		auto vec_h = obj.vecs_dict[current_level].get_access<cl::sycl::access::mode::write>(h);
		auto temp_vec = obj.temp_dict[current_level].get_access < cl::sycl::access::mode::read>(h);
		h.parallel_for(cl::sycl::range<1>{obj.num_dofs_per_level[current_level]}, [=](cl::sycl::id<1>idx) {
			vec_h[idx[0]] = temp_vec[idx[0]];
		});		
	});
	q.wait();
	if (current_level != finest_level) {

		for (int i = 0; i <= mu0_1; i++) {
			vcyclemultigrid(q, obj, current_level);
		}
	}
	else {
		for (int i = 0; i <= mu0; i++) {
			vcyclemultigrid(q, obj, current_level);
		}
	}
}

void solve(ProblemVar& obj) {
	cl::sycl::queue q;
	coarse_matrix_assemble(q, obj); // Assemble the Eigen Matrix for the coarsest level
	fullmultigrid(q, obj , finest_level );
	//return answer;
}
