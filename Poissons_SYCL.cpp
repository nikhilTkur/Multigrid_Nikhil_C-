#include <iostream>
#include <CL/sycl.hpp>
#include <vector>
#include <numeric>
#include "oneapi/mkl.hpp"
#include "mkl.h"
#include "mkl_spblas.h"
#include "mkl_graph.h"
#include <cmath>
#include <tuple>
#include <unordered_set>


using namespace sycl;
using namespace oneapi::mkl::sparse;

struct Matrix_Elements_for_Jacobi
{
	matrix_handle_t A_LU_handle;
	matrix_handle_t A_D_handle;
	matrix_handle_t R_omega_handle;
	int size;

};


// Defining the parameters of MultiGrid.
const int finest_level = 6;
const int coarsest_level = 3;
int highest_size = int(std::pow(2, finest_level));
const int mu0 = 30;
const int mu1 = 15;
const int mu2 = 15;

struct csr_data
{
	std::vector<float> data;
	std::vector<std::int32_t> indptr;
	std::vector<std::int32_t> indices;
};

// TODO: Parallel Kernel
csr_data coo_to_csr(const std::int32_t nrows, const std::int32_t ncols,
	const std::int32_t nnz, std::vector<std::int32_t> coo_row,
	std::vector<std::int32_t> coo_col, std::vector<float> coo_data)
{
	std::vector<float> data(nnz);
	std::vector<std::int32_t> indptr(nrows + 1);
	std::vector<std::int32_t> indices(nnz);

	// Compute number of non-zero entries per row of A
	std::vector<std::int32_t> nnz_row(nrows);
	for (std::int32_t i = 0; i < nnz; i++)
		nnz_row[coo_row[i]]++;

	std::partial_sum(nnz_row.begin(), nnz_row.end(), indptr.begin() + 1,
		std::plus<int>());
	// std::exclusive_scan(nnz_row.begin(), nnz_row.end(), indptr.begin(), 0);

	indptr[0] = 0;
	indptr[nrows] = nnz;

	std::fill(nnz_row.begin(), nnz_row.end(), 0);
	for (std::int32_t i = 0; i < nnz; i++)
	{
		std::int32_t row = coo_row[i];
		std::int32_t pos = nnz_row[row] + indptr[row];

		data[pos] = coo_data[i];
		indices[pos] = coo_col[i];

		nnz_row[row]++;
	}

	return { data, indptr, indices };
}




//Defining Global Matrices;
std::vector <Matrix_Elements_for_Jacobi> jacobi_matrices(finest_level - coarsest_level + 1); // Matrices along with their sizes (sizes to be used for V Cycle and FMG Cycle)

//Define the Domain size of the problem
const float domain_x = 1.0;
const float domain_y = 1.0;

//Define the force vector of the Poissons Equation
const float f = 4.0;

std::vector<float> JacobiRelaxation(matrix_handle_t R_omega,matrix_handle_t A_D, int A_size, std::vector<float> &v, std::vector<float>&f, const int &mu , cl::sycl::queue &q) {
	//
	const float omega = 2 / 3;
	std::vector<float> v_temp(v.size(), 0);
	std::vector<float> f_temp(v.size(), 0);
	cl::sycl::event v_temp_done = cl::sycl::event();
	cl::sycl::event f_temp_done = cl::sycl::event();
	cl::sycl::event final_add_done = cl::sycl::event();
	for (int i = 0; i < mu; i++) {
		v_temp_done = gemv(q, oneapi::mkl::transpose::nontrans, 1.0, R_omega, v.data(), 0.0, v_temp.data(), {});
		f_temp_done = gemv(q, oneapi::mkl::transpose::nontrans, omega / 4, A_D, f.data(), 0.0, f_temp.data(), {});
		final_add_done = oneapi::mkl::vm::add(q, f.size(), v_temp.data(), f_temp.data(), v.data(), { v_temp_done , f_temp_done });
		final_add_done.wait();
	}
	return v;
}

//using namespace oneapi::mkl::sparse;
std::vector<float> triangle_element_stiffness_matrix(float &x1, float  &x2, float  &x3, float  &y1, float  &y2, float  &y3){
	float aplha_1 = x2 * y3 - x3 * y2;
	float beta_1 = y2 - y3;
	float gamma_1 = -1 * (x2 - x3);
	float aplha_2 = x3 * y1 - x1 * y3;
	float beta_2 = y3 - y1;
	float gamma_2 = -1 * (x3 - x1);
	float aplha_3 = x1 * y2 - x2 * y1;
	float beta_3 = y1 - y2;
	float gamma_3 = -1 * (x1 - x2);
	float Ae = (aplha_1 + aplha_2 + aplha_3) / 2.0;    // Calculating the area of the triangular element

	std::vector <float> K_element(9, 0.0);
	K_element[0] = (beta_1 * beta_1 + gamma_1 * gamma_1) / (4 * Ae); //row 1
	K_element[1] = (beta_1 * beta_2 + gamma_1 * gamma_2) / (4 * Ae);
	K_element[2] = (beta_1 * beta_3 + gamma_1 * gamma_3) / (4 * Ae);
	K_element[3] = (beta_2 * beta_1 + gamma_2 * gamma_1) / (4 * Ae); //row2
	K_element[4] = (beta_2 * beta_2 + gamma_2 * gamma_2) / (4 * Ae);
	K_element[5] = (beta_2 * beta_3 + gamma_2 * gamma_3) / (4 * Ae);
	K_element[6] = (beta_3 * beta_1 + gamma_3 * gamma_1) / (4 * Ae); //row 3
	K_element[7] = (beta_3 * beta_2 + gamma_3 * gamma_2) / (4 * Ae);
	K_element[8] = (beta_3 * beta_3 + gamma_3 * gamma_3) / (4 * Ae);

	return K_element;
}

std::vector <float> force_function_element(float &x1, float  &x2, float  &x3, float  &y1, float  &y2, float  &y3) {
	float aplha_1 = x2 * y3 - x3 * y2;
	//float beta_1 = y2 - y3;
	//float gamma_1 = -1 * (x2 - x3);
	float aplha_2 = x3 * y1 - x1 * y3;
	//float beta_2 = y3 - y1;
	//float gamma_2 = -1 * (x3 - x1);
	float aplha_3 = x1 * y2 - x2 * y1;
	//float beta_3 = y1 - y2;
	//float gamma_3 = -1 * (x1 - x2);
	float Ae = (aplha_1 + aplha_2 + aplha_3) / 2.0;    // Calculating the area of the triangular element

	std::vector<float> f_element(3,0);
	f_element[0] = f * Ae / 3;
	f_element[1] = f * Ae / 3;
	f_element[2] = f * Ae / 3;
	return f_element;
}

std::unordered_set<int> boundary_nodes_indices(const int& size, const int&sqrt_size) {			//passing the size of the martix as the input
	std::unordered_set<int> boundary_nodes;

	for (int i = 0; i <= sqrt_size; i++) {
		boundary_nodes.insert({i , size-1 -i});
	}
	for (int i = 2 * sqrt_size - 1; i <= size - sqrt_size - 1; i = i + sqrt_size) {
		boundary_nodes.insert({ i , i + 1 });
	}
	return boundary_nodes;
}

void GlobalStiffenssMatrix(int &mat_size, std::vector <int>& rows_LU, std::vector <int> &cols_LU, std::vector <float> &vals_LU,
	std::vector <int> &rows_D, std::vector <int> &cols_D, std::vector <float>& vals_D) 
{ // TODO
	int d_size = std::sqrt(mat_size) -1;
	std::unordered_set<int> boundary_node_indices = boundary_nodes_indices(mat_size, d_size + 1);
	float h = domain_x / d_size;

	//Looping for the odd elements in the domain
	int odd_elements = 1;
	for (int i = 1; i <= d_size; i++) {
		for (int j = 1; j <= 2 * d_size; j = j + 2) {
			float x1 = ((j - 1) * h) / 2;
			float x2 = x1;
			float x3 = ((j + 1) * h) / 2;
			float y1 = (i - 1) * h;
			float y2 = i * h;
			float y3 = y2;
			std::vector<float>Ke = triangle_element_stiffness_matrix(x1, x2, x3, y1, y2, y3);
			std::vector <int> row_indices_global = {(odd_elements+1)/2 + i -2 ,(odd_elements + 1) / 2 + i - 2 , (odd_elements + 1) / 2 + i - 2 , (odd_elements + 1) / 2 + i + d_size - 1 ,(odd_elements + 1) / 2 + i + d_size - 1 ,(odd_elements + 1) / 2 + i + d_size - 1, (odd_elements + 1) / 2 + i + d_size ,(odd_elements + 1) / 2 + i + d_size ,(odd_elements + 1) / 2 + i + d_size };
			std::vector <int> col_indices_global = { (odd_elements + 1) / 2 + i -2, (odd_elements + 1)/2 + i + d_size-1 , (odd_elements + 1) / 2 + i + d_size ,(odd_elements + 1) / 2 + i - 2, (odd_elements + 1) / 2 + i + d_size - 1 , (odd_elements + 1) / 2 + i + d_size ,(odd_elements + 1) / 2 + i - 2, (odd_elements + 1) / 2 + i + d_size - 1 , (odd_elements + 1) / 2 + i + d_size };
			
			// loop to only include the non boundary elements in the stiffness matrix
			
			for (int k = 0; k < 9; k++) { 
				if (boundary_node_indices.find(row_indices_global[k]) == boundary_node_indices.end() && boundary_node_indices.find(col_indices_global[k]) == boundary_node_indices.end()) {
					//check if the entry is on the diagonal or LU
					if (row_indices_global[k] == col_indices_global[k]) {
						rows_D.push_back(row_indices_global[k] - (d_size + 2) - std::floor((row_indices_global[k] - (d_size + 2)) / (d_size + 1)) * 2);
						cols_D.push_back(col_indices_global[k] - (d_size + 2) - std::floor((col_indices_global[k] - (d_size + 2)) / (d_size + 1)) * 2);
						vals_D.push_back(Ke[k]);
					}
					else {
						rows_LU.push_back(row_indices_global[k] - (d_size + 2) - std::floor((row_indices_global[k] - (d_size + 2)) / (d_size + 1)) * 2);
						cols_LU.push_back(col_indices_global[k] - (d_size + 2) - std::floor((col_indices_global[k] - (d_size + 2)) / (d_size + 1)) * 2);
						vals_LU.push_back(Ke[k]);
						
					}
				}
			}
			odd_elements = odd_elements + 2;


		}
	}

	//Looping for the even elements in the domain
	int even_elements = 2;	
	for (int i = 1; i <= d_size; i++) {
		for (int j = 2; j <= 2 * d_size; j = j + 2) {
			float x1 = (j * h) / 2;
			float x2 = x1;
			float x3 = ((j - 2) * h) / 2;
			float y1 = i * h;
			float y2 = (i-1) * h;
			float y3 = y2;
			std::vector<float>Ke = triangle_element_stiffness_matrix(x1, x2, x3, y1, y2, y3);
			std::vector <int> row_indices_global = { even_elements / 2 + i + d_size , even_elements / 2 + i + d_size, even_elements / 2 + i + d_size , even_elements / 2 + i - 1 ,even_elements / 2 + i - 1 , even_elements / 2 + i - 1 , even_elements / 2 + i -2 ,even_elements / 2 + i - 2 , even_elements / 2 + i - 2 };
			std::vector <int> col_indices_global = { even_elements / 2 + i + d_size , even_elements / 2 + i - 1 , even_elements / 2 + i - 2 ,even_elements / 2 + i + d_size , even_elements / 2 + i - 1 , even_elements / 2 + i - 2 ,even_elements / 2 + i + d_size , even_elements / 2 + i - 1 , even_elements / 2 + i - 2 };


			// loop to only include the non boundary elements in the stiffness matrix

			for (int k = 0; k < 9; k++) {
				if (boundary_node_indices.find(row_indices_global[k]) == boundary_node_indices.end() && boundary_node_indices.find(col_indices_global[k]) == boundary_node_indices.end()) {
					//check if the entry is on the diagonal or LU
					if (row_indices_global[k] == col_indices_global[k]) {
						rows_D.push_back(row_indices_global[k] - (d_size + 2) - std::floor((row_indices_global[k] - (d_size + 2)) / (d_size + 1)) * 2);
						cols_D.push_back(col_indices_global[k] - (d_size + 2) - std::floor((col_indices_global[k] - (d_size + 2)) / (d_size + 1)) * 2);
						vals_D.push_back(Ke[k]);
					}
					else {
						rows_LU.push_back(row_indices_global[k] - (d_size + 2) - std::floor((row_indices_global[k] - (d_size + 2)) / (d_size + 1)) * 2);
						cols_LU.push_back(col_indices_global[k] - (d_size + 2) - std::floor((col_indices_global[k] - (d_size + 2)) / (d_size + 1)) * 2);
						vals_LU.push_back(Ke[k]);

					}
				}
			}
			even_elements = even_elements + 2;
		}
	}
}

std::vector <float> GlobalForceFunction() {
	//int sqrt_global_vector_size = int(std::pow(2, finest_level)) + 1;


	int nodes_per_row = highest_size + 1;
	int global_vector_size = nodes_per_row * nodes_per_row;
	std::unordered_set<int> boundary_node_indices_finest = boundary_nodes_indices(global_vector_size, nodes_per_row); //To check if the node entry in the global force vector is a boundary one or not. If boundary, it is skipped
	float h = domain_x / (highest_size);
	
	std::vector<float> global_force_vector(global_vector_size - 4*nodes_per_row + 4 , 0.0);
	int odd_element = 1;
	for (int i = 1; i <= highest_size; i++) {

		for (int j = 1; j <= 2 * highest_size; j = j + 2) {

			float x1 = ((j - 1) * h) / 2;
			float x2 = x1;
			float x3 = ((j + 1) * h) / 2;
			float y1 = (i - 1) * h;
			float y2 = i * h;
			float y3 = y2;
			std::vector<float> fe = force_function_element(x1, x2, x3, y1, y2, y3);
			std::vector<int> node_indices_finest = { (odd_element + 1) / 2 + i - 2 ,(odd_element + 1) / 2 + i + highest_size - 1,(odd_element + 1) / 2 + highest_size + i };
			for (int k = 0; k < 3; k++) {
				if (boundary_node_indices_finest.find(node_indices_finest[k]) == boundary_node_indices_finest.end()) {
					global_force_vector[node_indices_finest[k] - (highest_size + 2) - std::floor((node_indices_finest[k] - (highest_size + 2)) / (highest_size + 1)) * 2] += fe[k];
				}
			}
			odd_element += 2;
		}
	}
	int even_element = 2;
	for (int i = 1; i <= highest_size; i++) {

		for (int j = 2; j <= 2 * highest_size; j = j + 2) {

			float x1 = (j * h) / 2;
			float x2 = x1;
			float x3 = ((j - 2) * h) / 2;
			float y1 = i * h;
			float y2 = (i - 1) * h;
			float y3 = y2;
			std::vector<float> fe = force_function_element(x1, x2, x3, y1, y2, y3);
			std::vector<int> node_indices_finest = { even_element / 2 + i + highest_size ,even_element / 2 + i - 1,even_element / 2 + i - 2 };
			for (int k = 0; k < 3; k++) {
				if (boundary_node_indices_finest.find(node_indices_finest[k]) == boundary_node_indices_finest.end()) {
					global_force_vector[node_indices_finest[k] - (highest_size + 2) - std::floor((node_indices_finest[k] - (highest_size + 2)) / (highest_size + 1)) * 2] += fe[k];
				}
			}
			even_element += 2;
		}
	}
	return global_force_vector;
}

std::vector<float> Interpolation2D(std::vector <float>& vec_2h) {
	int vec_2h_dim = int(std::sqrt(vec_2h.size()));
	int vec_h_dim = 2 * vec_2h_dim + 1;
	std::vector<float> vec_h(vec_h_dim * vec_h_dim, 0);
	vec_h[0] = 0.25 * vec_2h[0];
	vec_h[vec_h_dim * vec_h_dim - 1] = 0.25 * vec_2h[vec_2h_dim * vec_2h_dim - 1];
	vec_h[vec_h_dim - 1] =  0.25 * vec_2h[vec_2h_dim - 1];
	vec_h[vec_h_dim * (vec_h_dim - 1)] = 0.25 * vec_2h[vec_2h_dim * (vec_2h_dim - 1)];
	
	//Looping over the topmost boundary of vec_h
	for (int j_h = 2; j_h <= vec_h_dim - 1; j_h++) {
		//int dummy_j_h = j + 1;
		if (j_h % 2 == 0) {
			int j_2h = int(j_h / 2);
			vec_h[j_h - 1] = 0.5 * vec_2h[j_2h - 1];
		}
		else {
			int j_2h = int((j_h - 1) / 2);
			vec_h[j_h - 1] = 0.25 * (vec_2h[j_2h - 1] + vec_2h[j_2h]);
		}
	}
	//Looping for the leftmost boundary of vec_h
	for (int i_h = 2; i_h <= vec_h_dim - 1; i_h++) {
		if (i_h % 2 == 0) {
			int i_2h = int(i_h/2);
			vec_h[(i_h - 1) * vec_h_dim] = 0.5 * vec_2h[(i_2h - 1) * vec_2h_dim];
		}
		else {
			int i_2h = int((i_h -1) / 2);
			vec_h[(i_h - 1) * vec_h_dim] = 0.25 * (vec_2h[(i_2h - 1) * vec_2h_dim] + vec_2h[(i_2h)*vec_2h_dim]);
		}
	}
	//Looping for the bottommost boundary
	for (int j_h = 2; j_h <= vec_h_dim - 1; j_h++) {
		if (j_h % 2 == 0) {
			int j_2h = int(j_h/2);
			vec_h[vec_h_dim * (vec_h_dim - 1) + j_h - 1] = 0.5 * (vec_2h[vec_2h_dim * (vec_2h_dim -1) + j_2h -1]);
		}
		else {
			int j_2h = int((j_h - 1) / 2);
			vec_h[vec_h_dim * (vec_h_dim - 1) + j_h - 1] = 0.25 * (vec_2h[vec_2h_dim * (vec_2h_dim-1) + j_2h - 1] + vec_2h[vec_2h_dim * (vec_2h_dim - 1) + j_2h]);
		}
	}
	//Looping for the rightmost boundary
	for (int i_h = 2; i_h <= vec_h_dim - 1; i_h++) {
		if (i_h % 2 == 0) {
			int i_2h = int(i_h /2);
			vec_h[(i_h-1) * vec_h_dim + vec_h_dim -1] = 0.5 * vec_2h[(i_2h - 1) * vec_2h_dim + vec_2h_dim - 1];
		}
		else {
			int i_2h = int((i_h -1)/2);
			vec_h[(i_h -1)* vec_h_dim + vec_h_dim -1] = 0.25 * (vec_2h[(i_2h -1) * vec_2h_dim + vec_2h_dim-1] + vec_2h[i_2h * vec_2h_dim + vec_2h_dim -1]);
		}
	}

	//Looping for the remaining elements in the vec_h vector
	for (int i_h = 2; i_h <= vec_h_dim - 1; i_h++) {

		for (int j_h = 2; j_h <= vec_h_dim - 1; j_h++) {

			//Both Even
			if (i_h % 2 == 0 && j_h % 2 == 0) {
				int i_2h = int(i_h /2);
				int j_2h = int(j_h / 2);
				vec_h[(i_h - 1) * vec_h_dim + j_h - 1] = vec_2h[(i_2h - 1) * vec_2h_dim + j_2h - 1];
			}
			// i_h odd and j_h even
			else if (i_h % 2 == 1 && j_h % 2 == 0) {
				int i_2h = int((i_h - 1) / 2);
				int j_2h = int(j_h / 2);
				vec_h[(i_h - 1) * vec_h_dim + j_h - 1] = 0.5 * (vec_2h[(i_2h - 1) * vec_2h_dim + j_2h - 1] + vec_2h[i_2h * vec_2h_dim + j_2h - 1]);
			}
			// i_h even and j_h odd
			else if (i_h % 2 == 0 && j_h % 2 == 1) {
				int i_2h = int(i_h / 2);
				int j_2h = int((j_h - 1) / 2);
				vec_h[(i_h - 1) * vec_h_dim + j_h - 1] = 0.5 * (vec_2h[(i_2h - 1) * vec_2h_dim + j_2h - 1] + vec_2h[(i_2h - 1) * vec_2h_dim + j_2h]);
			}
			// Both odd
			else {
				int i_2h = int((i_h - 1) / 2);
				int j_2h = int((j_h - 1) / 2);
				vec_h[(i_h - 1) * vec_h_dim + j_h - 1] = 0.25 * (vec_2h[(i_2h - 1) * vec_2h_dim + j_2h - 1] + vec_2h[i_2h * vec_2h_dim + j_2h - 1] + vec_2h[(i_2h - 1) * vec_2h_dim + j_2h] + vec_2h[i_2h * vec_2h_dim + j_2h]);
			}

		}
	}

	return vec_h;
}

std::vector <float> Restriction2D(std::vector <float>& vec_h) {
	int vec_h_dim = int(std::sqrt(vec_h.size()));
	int vec_2h_dim = int((vec_h_dim - 1) / 2);
	std::vector<float> vec_2h(vec_2h_dim * vec_2h_dim, 0);
	for (int i_2h = 1; i_2h <= vec_2h_dim; i_2h++) {

		for (int j_2h = 1; j_2h <= vec_2h_dim; j_2h++) {

			vec_2h[(i_2h - 1) * vec_2h_dim + j_2h - 1] = (1 / 16) * (vec_h[(2 * i_2h - 1 - 1) * vec_h_dim + 2 * j_2h - 1 - 1] + vec_h[(2 * i_2h - 1 - 1) * vec_h_dim + 2 * j_2h]
				+ vec_h[2 * i_2h * vec_h_dim + 2 * j_2h - 1 - 1] + vec_h[2 * i_2h * vec_h_dim + 2 * j_2h] + 2 * (vec_h[(2 * i_2h - 1) * vec_h_dim + 2 * j_2h - 1 - 1] +
					vec_h[(2 * i_2h - 1) * vec_h_dim + 2 * j_2h] + vec_h[(2 * i_2h - 1 - 1) * vec_h_dim + 2 * j_2h - 1] + vec_h[2 * i_2h * vec_h_dim + 2 * j_2h - 1]) +
				4 * vec_h[(2 * i_2h - 1) * vec_h_dim + 2 * j_2h - 1]);
		}
	}
	return vec_2h;
}

std::vector<float> VCycleMultigrid(Matrix_Elements_for_Jacobi &A_h, std::vector<float> &vec_h, std::vector<float> &f_h , sycl::queue &q) {

	matrix_handle_t R_omega_handle = A_h.R_omega_handle;
	matrix_handle_t A_D_handle = A_h.A_D_handle;
	int A_size = A_h.size;
	std::vector<float> vec_2h;
	vec_h = JacobiRelaxation(R_omega_handle,A_D_handle, A_size, vec_h, f_h, mu1 , q);

	if (int(std::log2(std::sqrt(A_size) + 1)) == coarsest_level) {

		vec_h = JacobiRelaxation(R_omega_handle, A_D_handle, A_size, vec_h, f_h, mu2,q);
		return vec_h;
	}
	else {
		//calculating the residual for the current solution
		std::vector<float> temp_vec1(vec_h.size(), 0);
		std::vector<float> temp_vec2(vec_h.size(), 0);
		std::vector<float> result(vec_h.size(), 0);
		std::vector<float> residual(f_h.size(), 0);
		std::vector<float> f_2h;
		const float a = 1.0;
		const float b = 0.0;
		//float* vec_ptr = vec_h.data();
		//float* result_ptr = result.data();
		cl::sycl::event gemv_LU_done = cl::sycl::event();
		cl::sycl::event gemv_D_done = cl::sycl::event();
		cl::sycl::event result_addn_done = cl::sycl::event();
		cl::sycl::event sub_done = cl::sycl::event();
		gemv_LU_done = gemv(q, oneapi::mkl::transpose::nontrans, a, A_h.A_LU_handle, vec_h.data(), b, temp_vec1.data(), {});
		gemv_D_done = gemv(q, oneapi::mkl::transpose::nontrans, a, A_D_handle, vec_h.data(), b, temp_vec2.data(), {});
		result_addn_done = oneapi::mkl::vm::add(q, vec_h.size(), temp_vec1.data(), temp_vec2.data(), result.data(), { gemv_LU_done , gemv_D_done });
		sub_done = oneapi::mkl::vm::sub(q, f_h.size(), f_h.data(), result.data(), residual.data(), { result_addn_done });
		sub_done.wait();
		
		//Applying the restriction on the residual
		f_2h = Restriction2D(residual);
		vec_2h = std::vector<float> (f_2h.size(), 0);

		//Fetching the Coarser Level Matrix
		Matrix_Elements_for_Jacobi A_2h = jacobi_matrices[int(std::log2(std::sqrt(f_2h.size()) + 1)) - coarsest_level];
		vec_2h = VCycleMultigrid(A_2h, vec_2h, f_2h, q);
	}
	std::vector<float> vec_2h_interpolation = Interpolation2D(vec_2h);
	cl::sycl::event vec_h_addition = cl::sycl::event();
	std::vector<float> vec_h_final(vec_h.size(), 0);
	vec_h_addition = oneapi::mkl::vm::add(q, vec_h.size(), vec_h.data(), vec_2h_interpolation.data(), vec_h_final.data(), {});
	vec_h_addition.wait();
	vec_h_final = JacobiRelaxation(R_omega_handle, A_D_handle, A_size, vec_h, f_h, mu2,q);
	return vec_h_final;
}

std::vector<float> FullMultiGrid(Matrix_Elements_for_Jacobi& A_h, std::vector<float>& f_h, sycl::queue& q) {
	std::vector<float> vec_h(f_h.size(), 0);
	std::vector<float> vec_2h;
	//matrix_handle_t A_handle = std::get<0>(A_h);
	int A_size = A_h.size;
	if (int(std::log2(std::sqrt(A_size) + 1)) == coarsest_level) {
		for (int i = 0; i <= mu0; i++) {
			vec_h = VCycleMultigrid(A_h, vec_h, f_h, q);
		}
		return vec_h;
	}
	else {
		std::vector<float> f_2h = Restriction2D(f_h);
		Matrix_Elements_for_Jacobi A_2h = jacobi_matrices[int(std::log2(std::sqrt(f_2h.size()) + 1)) - coarsest_level];
		vec_2h = FullMultiGrid(A_2h , f_2h , q);
	}
	vec_h = Interpolation2D(vec_2h);
	for (int i = 0; i <= mu0; i++) {
		vec_h = VCycleMultigrid(A_h, vec_h, f_h, q);
	}
	return vec_h;
	//Check if the level of the matrix is coarsest level
}

int main() {

	matrix_handle_t* handle;
	std::vector <matrix_handle_t*> vec;

	//Obtain the global matrices


		//


	//obtain the global force function

	//obtain the 

	//Do prifiling


		//convert the matrices in coo formats and transfer its handles to the FMG cycle

}

	


