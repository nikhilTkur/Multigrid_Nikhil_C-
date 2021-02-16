#include <iostream>
#include <CL/sycl.hpp>
#include <vector>
#include <numeric>
#include "oneapi/mkl.hpp"
// #include "mkl.h"
// #include "mkl_spblas.h"
#include <cmath>
#include <unordered_set>
#include <cstdint>


using namespace sycl;
using namespace oneapi::mkl::sparse;

// defining the parameters of multigrid.
const int finest_level = 10;
const int coarsest_level = 7;
std::int32_t highest_size = std::int32_t(std::pow(2, finest_level));
const int mu0 = 30;
const int mu1 = 10;
const int mu2 = 10;

struct matrix_elements_for_jacobi
{
	matrix_handle_t a_d_handle;
	//std::vector<float> a_diag;
	matrix_handle_t a_lu_handle;
	std::int32_t size;
};

//defining global matrices;
std::vector <matrix_elements_for_jacobi> jacobi_matrices(finest_level - coarsest_level + 1); // matrices along with their sizes (sizes to be used for v cycle and fmg cycle)

struct mat_coo_data {
	std::vector <std::int32_t> rows_lu;
	std::vector <std::int32_t> cols_lu;
	std::vector <float> vals_lu;
	std::vector <std::int32_t> rows_d;
	std::vector <std::int32_t> cols_d;
	std::vector <float> vals_d;
};
std::vector<mat_coo_data> global_matrices_coo_data(finest_level - coarsest_level + 1);

struct csr_data
{
	std::vector<float> data;
	std::vector<std::int32_t> indptr;
	std::vector<std::int32_t> indices;
};
std::vector<csr_data> global_matrices_csr_data_lu(finest_level - coarsest_level + 1);
std::vector<csr_data> global_matrices_csr_data_d(finest_level - coarsest_level + 1);

// todo: parallel kernel
csr_data coo_to_csr(const std::int32_t nrows, const std::int32_t ncols,
	const std::int32_t nnz, std::vector<std::int32_t> coo_row,
	std::vector<std::int32_t> coo_col, std::vector<float> coo_data)
{
	std::vector<float> data(nnz);
	std::vector<std::int32_t> indptr(nrows + 1);
	std::vector<std::int32_t> indices(nnz);

	// compute number of non-zero entries per row of a
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
	std::vector<float> data_csr;
	std::vector<std::int32_t>row_csr;
	int nnz_csr_new = indptr[0];
	row_csr.push_back(nnz_csr_new);
	std::vector<std::int32_t> col_csr;
	for (std::int32_t i = 0; i < indptr.size() - 1; i++) {
		std::int32_t col_no = indices[indptr[i]];
		std::int32_t data_str = data[indptr[i]];
		for (std::int32_t j = indptr[i] + 1; j < indptr[i + 1]; j++) {
			if (indices[j] == col_no) {
				data_str = data_str + data[j];
			}
			else {
				col_csr.push_back(col_no);
				data_csr.push_back(data_str);
				nnz_csr_new++;
				col_no = indices[j];
				data_str = data[j];
			}
		}
		if (indptr[i] < indices.size()) {
			col_csr.push_back(col_no);
			data_csr.push_back(data_str);
			nnz_csr_new++;
			row_csr.push_back(nnz_csr_new);
		}
	}
	//indptr[indptr.size()-1] = nnz_csr_new;

	return { data_csr, row_csr, col_csr };
}

//define the domain size of the problem
const float domain_x = 1.0;
const float domain_y = 1.0;

//define the force vector of the poissons equation
const float f = 4.0;

std::vector<float> jacobirelaxation(cl::sycl::queue& q, matrix_handle_t a_lu, std::int32_t a_size, std::vector<float>& v, std::vector<float>& fh, const int& mu) {
	//cl::sycl::queue q;
	const float omega = 2.0 / 3.0;
	std::int32_t vec_size = v.size();
	std::vector<float> fh_temp = fh;
	std::vector <float> lu_v_temp(vec_size, 0.0);
	std::vector <float> first_addn(vec_size, 0.0);
	cl::sycl::event v_temp_done = cl::sycl::event();
	cl::sycl::event fh_temp_done = cl::sycl::event();
	cl::sycl::event lu_v_temp_done = cl::sycl::event();
	cl::sycl::event add_1_done = cl::sycl::event();
	cl::sycl::event add_2_done = cl::sycl::event();
	for (int i = 0; i < mu; i++) {
		lu_v_temp_done = gemv(q, oneapi::mkl::transpose::nontrans, (-1.0 * omega / 4.0), a_lu, v.data(), 0.0, lu_v_temp.data(), {});
		v_temp_done = oneapi::mkl::blas::column_major::scal(q, vec_size, (1.0 - omega), v.data(), 1, { lu_v_temp_done });
		fh_temp_done = oneapi::mkl::blas::column_major::scal(q, vec_size, (omega / 4.0), fh_temp.data(), 1, {});
		add_1_done = oneapi::mkl::vm::add(q, vec_size, v.data(), fh_temp.data(), first_addn.data(), { v_temp_done , fh_temp_done });
		add_2_done = oneapi::mkl::vm::add(q, vec_size, first_addn.data(), lu_v_temp.data(), v.data(), { add_1_done });
		add_2_done.wait();
		fh_temp = fh;
	}
	return v;
}

std::vector<float> triangle_element_stiffness_matrix(float& x1, float& x2, float& x3, float& y1, float& y2, float& y3) {
	float aplha_1 = x2 * y3 - x3 * y2;
	float beta_1 = y2 - y3;
	float gamma_1 = -1 * (x2 - x3);
	float aplha_2 = x3 * y1 - x1 * y3;
	float beta_2 = y3 - y1;
	float gamma_2 = -1 * (x3 - x1);
	float aplha_3 = x1 * y2 - x2 * y1;
	float beta_3 = y1 - y2;
	float gamma_3 = -1 * (x1 - x2);
	float ae = (aplha_1 + aplha_2 + aplha_3) / 2.0;    // calculating the area of the triangular element

	std::vector <float> k_element(9, 0.0);
	k_element[0] = (beta_1 * beta_1 + gamma_1 * gamma_1) / (4 * ae); //row 1
	k_element[1] = (beta_1 * beta_2 + gamma_1 * gamma_2) / (4 * ae);
	k_element[2] = (beta_1 * beta_3 + gamma_1 * gamma_3) / (4 * ae);
	k_element[3] = (beta_2 * beta_1 + gamma_2 * gamma_1) / (4 * ae); //row2
	k_element[4] = (beta_2 * beta_2 + gamma_2 * gamma_2) / (4 * ae);
	k_element[5] = (beta_2 * beta_3 + gamma_2 * gamma_3) / (4 * ae);
	k_element[6] = (beta_3 * beta_1 + gamma_3 * gamma_1) / (4 * ae); //row 3
	k_element[7] = (beta_3 * beta_2 + gamma_3 * gamma_2) / (4 * ae);
	k_element[8] = (beta_3 * beta_3 + gamma_3 * gamma_3) / (4 * ae);

	return k_element;
}

std::vector <float> force_function_element(float& x1, float& x2, float& x3, float& y1, float& y2, float& y3) {
	float aplha_1 = x2 * y3 - x3 * y2;
	float aplha_2 = x3 * y1 - x1 * y3;
	float aplha_3 = x1 * y2 - x2 * y1;
	float ae = (aplha_1 + aplha_2 + aplha_3) / 2.0;    // calculating the area of the triangular element

	std::vector<float> f_element(3, 0);
	f_element[0] = f * ae / 3;
	f_element[1] = f * ae / 3;
	f_element[2] = f * ae / 3;
	return f_element;
}

std::unordered_set<std::int32_t> boundary_nodes_indices(const std::int32_t& size, const std::int32_t& sqrt_size) { // CORRECT
	std::unordered_set<std::int32_t> boundary_nodes;

	for (std::int32_t i = 0; i <= sqrt_size; i++) {
		boundary_nodes.insert({ i , size - 1 - i });
	}
	for (std::int32_t i = 2 * sqrt_size - 1; i <= size - sqrt_size - 1; i = i + sqrt_size) {
		boundary_nodes.insert({ i , i + 1 });
	}
	return boundary_nodes;
}

void globalstiffenssmatrix(std::int32_t& mat_size, std::vector <std::int32_t>& rows_lu, std::vector <std::int32_t>& cols_lu, std::vector <float>& vals_lu,
	std::vector <std::int32_t>& rows_d, std::vector <std::int32_t>& cols_d, std::vector <float>& vals_d)
{
	std::int32_t d_size = std::sqrt(mat_size) - 1;
	std::unordered_set<std::int32_t> boundary_node_indices = boundary_nodes_indices(mat_size, d_size + 1);
	float h = domain_x / d_size;

	//looping for the odd elements in the domain
	std::int32_t odd_elements = 1;
	for (std::int32_t i = 1; i <= d_size; i++) {
		for (std::int32_t j = 1; j <= 2 * d_size; j = j + 2) {
			float x1 = ((j - 1) * h) / 2;
			float x2 = x1;
			float x3 = ((j + 1) * h) / 2;
			float y1 = (i - 1) * h;
			float y2 = i * h;
			float y3 = y2;
			std::vector<float>ke = triangle_element_stiffness_matrix(x1, x2, x3, y1, y2, y3);
			std::vector <std::int32_t> row_indices_global = { (odd_elements + 1) / 2 + i - 2 ,(odd_elements + 1) / 2 + i - 2 , (odd_elements + 1) / 2 + i - 2 , (odd_elements + 1) / 2 + i + d_size - 1 ,(odd_elements + 1) / 2 + i + d_size - 1 ,(odd_elements + 1) / 2 + i + d_size - 1, (odd_elements + 1) / 2 + i + d_size ,(odd_elements + 1) / 2 + i + d_size ,(odd_elements + 1) / 2 + i + d_size };
			std::vector <std::int32_t> col_indices_global = { (odd_elements + 1) / 2 + i - 2, (odd_elements + 1) / 2 + i + d_size - 1 , (odd_elements + 1) / 2 + i + d_size ,(odd_elements + 1) / 2 + i - 2, (odd_elements + 1) / 2 + i + d_size - 1 , (odd_elements + 1) / 2 + i + d_size ,(odd_elements + 1) / 2 + i - 2, (odd_elements + 1) / 2 + i + d_size - 1 , (odd_elements + 1) / 2 + i + d_size };

			// loop to only include the non boundary elements in the stiffness matrix

			for (std::int32_t k = 0; k < 9; k++) {
				if (boundary_node_indices.find(row_indices_global[k]) == boundary_node_indices.end() && boundary_node_indices.find(col_indices_global[k]) == boundary_node_indices.end()) {
					//check if the entry is on the diagonal or lu
					if (row_indices_global[k] == col_indices_global[k]) {
						rows_d.push_back(row_indices_global[k] - (d_size + 2) - std::floor((row_indices_global[k] - (d_size + 2)) / (d_size + 1)) * 2);
						cols_d.push_back(col_indices_global[k] - (d_size + 2) - std::floor((col_indices_global[k] - (d_size + 2)) / (d_size + 1)) * 2);
						vals_d.push_back(ke[k]);
					}
					else {
						rows_lu.push_back(row_indices_global[k] - (d_size + 2) - std::floor((row_indices_global[k] - (d_size + 2)) / (d_size + 1)) * 2);
						cols_lu.push_back(col_indices_global[k] - (d_size + 2) - std::floor((col_indices_global[k] - (d_size + 2)) / (d_size + 1)) * 2);
						vals_lu.push_back(ke[k]);

					}
				}
			}
			odd_elements = odd_elements + 2;


		}
	}

	//looping for the even elements in the domain
	std::int32_t even_elements = 2;
	for (std::int32_t i = 1; i <= d_size; i++) {
		for (std::int32_t j = 2; j <= 2 * d_size; j = j + 2) {
			float x1 = (j * h) / 2;
			float x2 = x1;
			float x3 = ((j - 2) * h) / 2;
			float y1 = i * h;
			float y2 = (i - 1) * h;
			float y3 = y2;
			std::vector<float>ke = triangle_element_stiffness_matrix(x1, x2, x3, y1, y2, y3);
			std::vector <std::int32_t> row_indices_global = { even_elements / 2 + i + d_size , even_elements / 2 + i + d_size, even_elements / 2 + i + d_size , even_elements / 2 + i - 1 ,even_elements / 2 + i - 1 , even_elements / 2 + i - 1 , even_elements / 2 + i - 2 ,even_elements / 2 + i - 2 , even_elements / 2 + i - 2 };
			std::vector <std::int32_t> col_indices_global = { even_elements / 2 + i + d_size , even_elements / 2 + i - 1 , even_elements / 2 + i - 2 ,even_elements / 2 + i + d_size , even_elements / 2 + i - 1 , even_elements / 2 + i - 2 ,even_elements / 2 + i + d_size , even_elements / 2 + i - 1 , even_elements / 2 + i - 2 };


			// loop to only include the non boundary elements in the stiffness matrix

			for (std::int32_t k = 0; k < 9; k++) {
				if (boundary_node_indices.find(row_indices_global[k]) == boundary_node_indices.end() && boundary_node_indices.find(col_indices_global[k]) == boundary_node_indices.end()) {
					//check if the entry is on the diagonal or lu
					if (row_indices_global[k] == col_indices_global[k]) {
						rows_d.push_back(row_indices_global[k] - (d_size + 2) - std::floor((row_indices_global[k] - (d_size + 2)) / (d_size + 1)) * 2);
						cols_d.push_back(col_indices_global[k] - (d_size + 2) - std::floor((col_indices_global[k] - (d_size + 2)) / (d_size + 1)) * 2);
						vals_d.push_back(ke[k]);
					}
					else {
						rows_lu.push_back(row_indices_global[k] - (d_size + 2) - std::floor((row_indices_global[k] - (d_size + 2)) / (d_size + 1)) * 2);
						cols_lu.push_back(col_indices_global[k] - (d_size + 2) - std::floor((col_indices_global[k] - (d_size + 2)) / (d_size + 1)) * 2);
						vals_lu.push_back(ke[k]);

					}
				}
			}
			even_elements = even_elements + 2;
		}
	}
}

std::vector <float> globalforcefunction() {
	//int sqrt_global_vector_size = int(std::pow(2, finest_level)) + 1;

	std::int32_t nodes_per_row = highest_size + 1;
	std::int32_t global_vector_size = nodes_per_row * nodes_per_row;
	std::unordered_set<std::int32_t> boundary_node_indices_finest = boundary_nodes_indices(global_vector_size, nodes_per_row); //to check if the node entry in the global force vector is a boundary one or not. if boundary, it is skipped
	float h = domain_x / (highest_size);

	std::vector<float> global_force_vector(global_vector_size - 4 * nodes_per_row + 4, 0.0);
	std::int32_t odd_element = 1;
	for (std::int32_t i = 1; i <= highest_size; i++) {

		for (std::int32_t j = 1; j <= 2 * highest_size; j = j + 2) {

			float x1 = ((j - 1) * h) / 2;
			float x2 = x1;
			float x3 = ((j + 1) * h) / 2;
			float y1 = (i - 1) * h;
			float y2 = i * h;
			float y3 = y2;
			std::vector<float> fe = force_function_element(x1, x2, x3, y1, y2, y3);
			std::vector<std::int32_t> node_indices_finest = { (odd_element + 1) / 2 + i - 2 ,(odd_element + 1) / 2 + i + highest_size - 1,(odd_element + 1) / 2 + highest_size + i };
			for (std::int32_t k = 0; k < 3; k++) {
				if (boundary_node_indices_finest.find(node_indices_finest[k]) == boundary_node_indices_finest.end()) {
					global_force_vector[node_indices_finest[k] - (highest_size + 2) - std::floor((node_indices_finest[k] - (highest_size + 2)) / (highest_size + 1)) * 2] += fe[k];
				}
			}
			odd_element += 2;
		}
	}
	std::int32_t even_element = 2;
	for (std::int32_t i = 1; i <= highest_size; i++) {

		for (std::int32_t j = 2; j <= 2 * highest_size; j = j + 2) {

			float x1 = (j * h) / 2;
			float x2 = x1;
			float x3 = ((j - 2) * h) / 2;
			float y1 = i * h;
			float y2 = (i - 1) * h;
			float y3 = y2;
			std::vector<float> fe = force_function_element(x1, x2, x3, y1, y2, y3);
			std::vector<std::int32_t> node_indices_finest = { even_element / 2 + i + highest_size ,even_element / 2 + i - 1,even_element / 2 + i - 2 };
			for (std::int32_t k = 0; k < 3; k++) {
				if (boundary_node_indices_finest.find(node_indices_finest[k]) == boundary_node_indices_finest.end()) {
					global_force_vector[node_indices_finest[k] - (highest_size + 2) - std::floor((node_indices_finest[k] - (highest_size + 2)) / (highest_size + 1)) * 2] += fe[k];
				}
			}
			even_element += 2;
		}
	}
	return global_force_vector;
}

std::vector<float> interpolation2d(std::vector <float>& vec_2h) {
	std::int32_t vec_2h_dim = std::int32_t(std::sqrt(vec_2h.size()));
	std::int32_t vec_h_dim = 2 * vec_2h_dim + 1;
	std::vector<float> vec_h(vec_h_dim * vec_h_dim, 0);
	vec_h[0] = 0.25 * vec_2h[0];
	vec_h[vec_h_dim * vec_h_dim - 1] = 0.25 * vec_2h[vec_2h_dim * vec_2h_dim - 1];
	vec_h[vec_h_dim - 1] = 0.25 * vec_2h[vec_2h_dim - 1];
	vec_h[vec_h_dim * (vec_h_dim - 1)] = 0.25 * vec_2h[vec_2h_dim * (vec_2h_dim - 1)];

	//looping over the topmost boundary of vec_h
	for (std::int32_t j_h = 2; j_h <= vec_h_dim - 1; j_h++) {
		//int dummy_j_h = j + 1;
		if (j_h % 2 == 0) {
			std::int32_t j_2h = j_h / 2;
			vec_h[j_h - 1] = 0.5 * vec_2h[j_2h - 1];
		}
		else {
			std::int32_t j_2h = (j_h - 1) / 2;
			vec_h[j_h - 1] = 0.25 * (vec_2h[j_2h - 1] + vec_2h[j_2h]);
		}
	}
	//looping for the leftmost boundary of vec_h
	for (int i_h = 2; i_h <= vec_h_dim - 1; i_h++) {
		if (i_h % 2 == 0) {
			std::int32_t i_2h = i_h / 2;
			vec_h[(i_h - 1) * vec_h_dim] = 0.5 * vec_2h[(i_2h - 1) * vec_2h_dim];
		}
		else {
			std::int32_t i_2h = (i_h - 1) / 2;
			vec_h[(i_h - 1) * vec_h_dim] = 0.25 * (vec_2h[(i_2h - 1) * vec_2h_dim] + vec_2h[(i_2h)*vec_2h_dim]);
		}
	}
	//looping for the bottommost boundary
	for (std::int32_t j_h = 2; j_h <= vec_h_dim - 1; j_h++) {
		if (j_h % 2 == 0) {
			std::int32_t j_2h = j_h / 2;
			vec_h[vec_h_dim * (vec_h_dim - 1) + j_h - 1] = 0.5 * (vec_2h[vec_2h_dim * (vec_2h_dim - 1) + j_2h - 1]);
		}
		else {
			std::int32_t j_2h = (j_h - 1) / 2;
			vec_h[vec_h_dim * (vec_h_dim - 1) + j_h - 1] = 0.25 * (vec_2h[vec_2h_dim * (vec_2h_dim - 1) + j_2h - 1] + vec_2h[vec_2h_dim * (vec_2h_dim - 1) + j_2h]);
		}
	}
	//looping for the rightmost boundary
	for (std::int32_t i_h = 2; i_h <= vec_h_dim - 1; i_h++) {
		if (i_h % 2 == 0) {
			std::int32_t i_2h = i_h / 2;
			vec_h[(i_h - 1) * vec_h_dim + vec_h_dim - 1] = 0.5 * vec_2h[(i_2h - 1) * vec_2h_dim + vec_2h_dim - 1];
		}
		else {
			std::int32_t i_2h = (i_h - 1) / 2;
			vec_h[(i_h - 1) * vec_h_dim + vec_h_dim - 1] = 0.25 * (vec_2h[(i_2h - 1) * vec_2h_dim + vec_2h_dim - 1] + vec_2h[i_2h * vec_2h_dim + vec_2h_dim - 1]);
		}
	}

	//looping for the remaining elements in the vec_h vector
	for (std::int32_t i_h = 2; i_h <= vec_h_dim - 1; i_h++) {

		for (std::int32_t j_h = 2; j_h <= vec_h_dim - 1; j_h++) {

			//both even
			if (i_h % 2 == 0 && j_h % 2 == 0) {
				std::int32_t i_2h = i_h / 2;
				std::int32_t j_2h = j_h / 2;
				vec_h[(i_h - 1) * vec_h_dim + j_h - 1] = vec_2h[(i_2h - 1) * vec_2h_dim + j_2h - 1];
			}
			// i_h odd and j_h even
			else if (i_h % 2 == 1 && j_h % 2 == 0) {
				std::int32_t i_2h = (i_h - 1) / 2;
				std::int32_t j_2h = j_h / 2;
				vec_h[(i_h - 1) * vec_h_dim + j_h - 1] = 0.5 * (vec_2h[(i_2h - 1) * vec_2h_dim + j_2h - 1] + vec_2h[i_2h * vec_2h_dim + j_2h - 1]);
			}
			// i_h even and j_h odd
			else if (i_h % 2 == 0 && j_h % 2 == 1) {
				std::int32_t i_2h = i_h / 2;
				std::int32_t j_2h = (j_h - 1) / 2;
				vec_h[(i_h - 1) * vec_h_dim + j_h - 1] = 0.5 * (vec_2h[(i_2h - 1) * vec_2h_dim + j_2h - 1] + vec_2h[(i_2h - 1) * vec_2h_dim + j_2h]);
			}
			// both odd
			else {
				std::int32_t i_2h = (i_h - 1) / 2;
				std::int32_t j_2h = (j_h - 1) / 2;
				vec_h[(i_h - 1) * vec_h_dim + j_h - 1] = 0.25 * (vec_2h[(i_2h - 1) * vec_2h_dim + j_2h - 1] + vec_2h[i_2h * vec_2h_dim + j_2h - 1] + vec_2h[(i_2h - 1) * vec_2h_dim + j_2h] + vec_2h[i_2h * vec_2h_dim + j_2h]);
			}

		}
	}
	return vec_h;
}

std::vector<float> interpolation2d_parallel(std::vector <float>& vec_2h) {
	queue q;
	size_t vec_2h_dim = size_t(std::sqrt(vec_2h.size()));
	size_t vec_h_dim = 2 * vec_2h_dim + 1;
	std::vector<float> vec_h(vec_h_dim * vec_h_dim, 0);
	vec_h[0] = 0.25 * vec_2h[0];
	vec_h[vec_h_dim * vec_h_dim - 1] = 0.25 * vec_2h[vec_2h_dim * vec_2h_dim - 1];
	vec_h[vec_h_dim - 1] = 0.25 * vec_2h[vec_2h_dim - 1];
	vec_h[vec_h_dim * (vec_h_dim - 1)] = 0.25 * vec_2h[vec_2h_dim * (vec_2h_dim - 1)];
	{
		buffer <float, 2> vec_2h_buf(vec_2h.data(), range<2>{vec_2h_dim, vec_2h_dim});
		buffer <float, 2> vec_h_buf(vec_h.data(), range<2>{vec_h_dim, vec_h_dim });
		q.submit([&](handler& h) {
			accessor vec_h_acc{ vec_h_buf, h };
			accessor vec_2h_acc{ vec_2h_buf , h };

			//domain has been divided such that the number of elements in each dimension is a power of 2. hence the number of nodes in each dim is odd.

			size_t num_even_per_dim = (vec_h_dim - 1) / 2;
			size_t num_odd_per_dim = num_even_per_dim - 1;

			//parallel loop for processing only the even numbered boundary nodes (index = node_number -1) of the structured domain

			h.parallel_for(range<1>{num_even_per_dim}, [=](id<1>idx) {
				size_t j_h = idx[0] * 2 + 2;
				//int j_2h = int(j_h / 2);
				vec_h_acc[0][j_h - 1] = 0.5 * vec_2h_acc[0][j_h / 2 - 1];								//top boundary nodes
				vec_h_acc[j_h - 1][0] = 0.5 * vec_2h_acc[j_h / 2 - 1][0];								//left boundary nodes
				vec_h_acc[vec_h_dim - 1][j_h - 1] = 0.5 * vec_2h_acc[vec_2h_dim - 1][j_h / 2 - 1];		// bottom boundary nodes
				vec_h_acc[j_h - 1][vec_h_dim - 1] = 0.5 * vec_2h_acc[j_h / 2 - 1][vec_2h_dim - 1];		//right boundary nodes
				});
			});

		//parallel loop for processing only the odd numbered boundary nodes (index = node_number -1) of the structured domain
		q.submit([&](handler& h) {
			accessor vec_h_acc{ vec_h_buf, h };
			accessor vec_2h_acc{ vec_2h_buf , h };
			size_t num_even_per_dim = (vec_h_dim - 1) / 2;
			size_t num_odd_per_dim = num_even_per_dim - 1;

			h.parallel_for(range<1>{num_odd_per_dim}, [=](id<1>idx) {
				size_t j_h = idx[0] * 2 + 3;
				size_t j_2h = (j_h - 1) / 2;
				vec_h_acc[0][j_h - 1] = 0.25 * (vec_2h_acc[0][j_2h - 1] + vec_2h_acc[0][j_2h - 1]);										//top boundary nodes
				vec_h_acc[j_h - 1][0] = 0.25 * (vec_2h_acc[j_2h - 1][0] + vec_2h_acc[j_2h - 1][0]);										//left boundary nodes
				vec_h_acc[vec_h_dim - 1][j_h - 1] = 0.25 * (vec_2h_acc[vec_2h_dim - 1][j_2h - 1] + vec_2h_acc[vec_2h_dim - 1][j_2h]);	//bottom boundary nodes
				vec_h_acc[j_h - 1][vec_h_dim - 1] = 0.25 * (vec_2h_acc[j_2h - 1][vec_2h_dim - 1] + vec_2h_acc[j_2h][vec_2h_dim - 1]);	//right boundary nodes
				});
			});

		//parallel loop for procesing even row even col nodes in the structured domain
		q.submit([&](handler& h) {
			accessor vec_h_acc{ vec_h_buf, h };
			accessor vec_2h_acc{ vec_2h_buf , h };
			size_t num_even_per_dim = (vec_h_dim - 1) / 2;
			size_t num_odd_per_dim = num_even_per_dim - 1;
			h.parallel_for(range<2>{num_even_per_dim, num_even_per_dim}, [=](id<2>idx) {
				size_t i_h = idx[0] * 2 + 2;
				size_t j_h = idx[1] * 2 + 2;
				vec_h_acc[i_h - 1][j_h - 1] = vec_2h_acc[i_h / 2 - 1][j_h / 2 - 1];
				});
			});
		q.submit([&](handler& h) {
			accessor vec_h_acc{ vec_h_buf, h };
			accessor vec_2h_acc{ vec_2h_buf , h };
			size_t num_even_per_dim = (vec_h_dim - 1) / 2;
			size_t num_odd_per_dim = num_even_per_dim - 1;
			h.parallel_for(range<2>{num_odd_per_dim, num_even_per_dim}, [=](id<2>idx) {		//odd row even col nodes in structured domain
				size_t i_h = idx[0] * 2 + 3;
				size_t j_h = idx[1] * 2 + 2;
				size_t i_2h = (i_h - 1) / 2;
				vec_h_acc[i_h - 1][j_h - 1] = 0.5 * (vec_2h_acc[i_2h - 1][j_h / 2 - 1] + vec_2h_acc[i_2h][j_h / 2 - 1]);
				});
			});
		q.submit([&](handler& h) {
			accessor vec_h_acc{ vec_h_buf, h };
			accessor vec_2h_acc{ vec_2h_buf , h };
			size_t num_even_per_dim = (vec_h_dim - 1) / 2;
			size_t num_odd_per_dim = num_even_per_dim - 1;
			h.parallel_for(range<2>{num_even_per_dim, num_odd_per_dim}, [=](id<2>idx) {		//even row odd cols nodes in structured domain
				size_t i_h = idx[0] * 2 + 2;
				size_t j_h = idx[1] * 2 + 3;
				size_t j_2h = (j_h - 1) / 2;
				vec_h_acc[i_h - 1][j_h - 1] = 0.5 * (vec_2h_acc[i_h / 2 - 1][j_2h - 1] + vec_2h_acc[i_h / 2 - 1][j_2h]);
				});
			});
		q.submit([&](handler& h) {
			accessor vec_h_acc{ vec_h_buf, h };
			accessor vec_2h_acc{ vec_2h_buf , h };
			size_t num_even_per_dim = (vec_h_dim - 1) / 2;
			size_t num_odd_per_dim = num_even_per_dim - 1;
			h.parallel_for(range<2>{num_odd_per_dim, num_odd_per_dim}, [=](id<2>idx) {		//both odd row and col nodes in structured domain
				size_t i_h = idx[0] * 2 + 3;
				size_t j_h = idx[1] * 2 + 3;
				size_t i_2h = (i_h - 1) / 2;
				size_t j_2h = (j_h - 1) / 2;
				vec_h_acc[i_h - 1][j_h - 1] = 0.25 * (vec_2h_acc[i_2h - 1][j_2h - 1] + vec_2h_acc[i_2h][j_2h - 1] + vec_2h_acc[i_2h - 1][j_2h] + vec_2h_acc[i_2h][j_2h]);
				});
			});
		q.wait();
	}
	return vec_h;
}

std::vector <float> restriction2d(std::vector <float>& vec_h) {
	std::int32_t vec_h_dim = std::sqrt(vec_h.size());
	std::int32_t vec_2h_dim = (vec_h_dim - 1) / 2;
	std::vector<float> vec_2h(vec_2h_dim * vec_2h_dim, 0);
	for (std::int32_t i_2h = 1; i_2h <= vec_2h_dim; i_2h++) {

		for (std::int32_t j_2h = 1; j_2h <= vec_2h_dim; j_2h++) {

			vec_2h[(i_2h - 1) * vec_2h_dim + j_2h - 1] = (1 / 16) * (vec_h[(2 * i_2h - 1 - 1) * vec_h_dim + 2 * j_2h - 1 - 1] + vec_h[(2 * i_2h - 1 - 1) * vec_h_dim + 2 * j_2h]
				+ vec_h[2 * i_2h * vec_h_dim + 2 * j_2h - 1 - 1] + vec_h[2 * i_2h * vec_h_dim + 2 * j_2h] + 2 * (vec_h[(2 * i_2h - 1) * vec_h_dim + 2 * j_2h - 1 - 1] +
					vec_h[(2 * i_2h - 1) * vec_h_dim + 2 * j_2h] + vec_h[(2 * i_2h - 1 - 1) * vec_h_dim + 2 * j_2h - 1] + vec_h[2 * i_2h * vec_h_dim + 2 * j_2h - 1]) +
				4 * vec_h[(2 * i_2h - 1) * vec_h_dim + 2 * j_2h - 1]);
		}
	}
	return vec_2h;
}

std::vector <float> restriction2d_parallel(std::vector <float>& vec_h) {
	std::size_t vec_h_dim = std::sqrt(vec_h.size());
	std::size_t vec_2h_dim = (vec_h_dim - 1) / 2;
	std::vector<float> vec_2h(vec_2h_dim * vec_2h_dim, 0);
	cl::sycl::queue q;
	{
		buffer <float, 2> vec_2h_buf(vec_2h.data(), range<2>{vec_2h_dim, vec_2h_dim});
		buffer <float, 2> vec_h_buf(vec_h.data(), range<2>{vec_h_dim, vec_h_dim});

		//float* host_vector_2h = malloc_host<float>(vec_2h_dim, q);
		q.submit([&](handler& h) {
			accessor vec_2h_acc{ vec_2h_buf , h };
			accessor vec_h_acc{ vec_h_buf , h };
			h.parallel_for(range<2>{ vec_2h_dim, vec_2h_dim}, [=](id<2>idx) {
				std::int32_t i_2h = idx[0]; //0 to vec_2h_dim -1
				std::int32_t j_2h = idx[1]; //0 to vec_2h_dim -1
				vec_2h_acc[i_2h][j_2h] = (1 / 16) * (vec_h_acc[2 * i_2h - 1][2 * j_2h - 1] + vec_h_acc[2 * i_2h - 1][2 * j_2h + 1]
					+ vec_h_acc[2 * i_2h + 1][2 * j_2h - 1] + vec_h_acc[2 * i_2h + 1][2 * j_2h + 1] + 2 * (vec_h_acc[2 * i_2h][2 * j_2h - 1] +
						vec_h_acc[2 * i_2h][2 * j_2h + 1] + vec_h_acc[2 * i_2h - 1][2 * j_2h] + vec_h_acc[2 * i_2h + 1][2 * j_2h]) +
					4 * vec_h_acc[2 * i_2h][2 * j_2h]);
				});
			});
		q.wait();
	}
	return vec_2h;
}

std::vector<float> vcyclemultigrid(cl::sycl::queue& q, matrix_elements_for_jacobi& a_h, std::vector<float>& vec_h, std::vector<float>& f_h) {
	//matrix_handle_t a_lu_hndl = a_h.a_lu_handle;
	//matrix_handle_t a_d_hndl = a_h.a_d_handle;
	std::int32_t a_size = a_h.size;
	std::int32_t vec_size = vec_h.size();
	std::vector<float> vec_2h;
	vec_h = jacobirelaxation(q, a_h.a_lu_handle, a_size, vec_h, f_h, mu1);

	if (int(std::log2(std::sqrt(a_size) + 1)) == coarsest_level) {

		vec_h = jacobirelaxation(q, a_h.a_lu_handle, a_size, vec_h, f_h, mu2);
		return vec_h;
	}
	else {
		//calculating the residual for the current solution
		//sycl::queue q;
		std::vector<float> temp_vec1(vec_size, 0);
		std::vector<float> temp_vec2(vec_size, 0);
		std::vector<float> result(vec_size, 0);
		std::vector<float> residual(vec_size, 0);
		std::vector<float> f_2h;
		const float a = 1.0;
		const float b = 0.0;
		//float* vec_ptr = vec_h.data();
		//float* result_ptr = result.data();
		cl::sycl::event gemv_lu_done = cl::sycl::event();
		cl::sycl::event gemv_d_done = cl::sycl::event();
		cl::sycl::event result_addn_done = cl::sycl::event();
		cl::sycl::event sub_done = cl::sycl::event();
		gemv_lu_done = gemv(q, oneapi::mkl::transpose::nontrans, a, a_h.a_lu_handle, vec_h.data(), b, temp_vec1.data(), {});
		gemv_d_done = gemv(q, oneapi::mkl::transpose::nontrans, a, a_h.a_d_handle, vec_h.data(), b, temp_vec2.data(), {});
		result_addn_done = oneapi::mkl::vm::add(q, vec_size, temp_vec1.data(), temp_vec2.data(), result.data(), { gemv_lu_done , gemv_d_done });
		sub_done = oneapi::mkl::vm::sub(q, vec_size, f_h.data(), result.data(), residual.data(), { result_addn_done });
		sub_done.wait();

		//applying the restriction on the residual
		f_2h = restriction2d(residual);
		std::int32_t f_2h_size = f_2h.size();
		vec_2h = std::vector<float>(f_2h_size, 0);

		//fetching the coarser level matrix
		matrix_elements_for_jacobi a_2h = jacobi_matrices[int(std::log2(std::sqrt(f_2h_size) + 1)) - coarsest_level];
		vec_2h = vcyclemultigrid(q, a_2h, vec_2h, f_2h);
	}
	//sycl::queue q;
	std::vector<float> vec_2h_interpolation = interpolation2d(vec_2h);
	cl::sycl::event vec_h_addition = cl::sycl::event();
	std::vector<float> vec_h_final(vec_size, 0);
	vec_h_addition = oneapi::mkl::vm::add(q, vec_size, vec_h.data(), vec_2h_interpolation.data(), vec_h_final.data(), {});
	vec_h_addition.wait();
	vec_h_final = jacobirelaxation(q, a_h.a_lu_handle, a_size, vec_h_final, f_h, mu2);
	return vec_h_final;
}

std::vector<float> fullmultigrid(cl::sycl::queue& q, matrix_elements_for_jacobi& a_h, std::vector<float>& f_h) {
	std::vector<float> vec_h(f_h.size(), 0);
	std::vector<float> vec_2h;
	//matrix_handle_t a_handle = std::get<0>(a_h);
	std::int32_t a_size = a_h.size;
	if (std::int32_t(std::log2(std::sqrt(a_size) + 1)) == coarsest_level) {
		for (std::int32_t i = 0; i <= mu0; i++) {
			vec_h = vcyclemultigrid(q, a_h, vec_h, f_h);
		}
		return vec_h;
	}
	else {
		std::vector<float> f_2h = restriction2d(f_h);
		matrix_elements_for_jacobi a_2h = jacobi_matrices[int(std::log2(std::sqrt(f_2h.size()) + 1)) - coarsest_level];
		vec_2h = fullmultigrid(q, a_2h, f_2h);
	}
	vec_h = interpolation2d(vec_2h);
	for (std::int32_t i = 0; i <= mu0; i++) {
		vec_h = vcyclemultigrid(q, a_h, vec_h, f_h);
	}
	return vec_h;
}

void inverse_diagonal(std::vector<float>& diag) {
	for (std::int32_t i = 0; i < diag.size(); i++) {
		diag[i] = 1 / diag[i];
	}
}

int main() {
	cl::sycl::queue q;
	//creating the global matrices for different levels.
	for (int level = coarsest_level; level <= finest_level; level++) {
		std::int32_t nodes_per_dim = std::pow(2, level) + 1;
		std::int32_t original_mat_size = nodes_per_dim * nodes_per_dim;
		std::int32_t non_bdry_mat_size = (nodes_per_dim - 2) * (nodes_per_dim - 2); //generating only the matrix of non boundary elements
		int index = level - coarsest_level;

		std::vector <std::int32_t> rows_lu;
		std::vector <std::int32_t> cols_lu;
		std::vector <float> vals_lu;
		std::vector <std::int32_t> rows_d;
		std::vector <std::int32_t> cols_d;
		std::vector <float> vals_d;

		globalstiffenssmatrix(original_mat_size, rows_lu, cols_lu, vals_lu, rows_d, cols_d, vals_d);

		global_matrices_csr_data_lu[index] = coo_to_csr(non_bdry_mat_size, non_bdry_mat_size, rows_lu.size(), rows_lu, cols_lu, vals_lu);

		global_matrices_csr_data_d[index] = coo_to_csr(non_bdry_mat_size, non_bdry_mat_size, rows_d.size(), rows_d, cols_d, vals_d);

		init_matrix_handle(&(jacobi_matrices[index].a_lu_handle));
		init_matrix_handle(&(jacobi_matrices[index].a_d_handle));

		set_csr_data(jacobi_matrices[index].a_lu_handle, non_bdry_mat_size, non_bdry_mat_size, oneapi::mkl::index_base::zero, global_matrices_csr_data_lu[index].indptr.data(), global_matrices_csr_data_lu[index].indices.data(), global_matrices_csr_data_lu[index].data.data());

		set_csr_data(jacobi_matrices[index].a_d_handle, non_bdry_mat_size, non_bdry_mat_size, oneapi::mkl::index_base::zero, global_matrices_csr_data_d[index].indptr.data(), global_matrices_csr_data_d[index].indices.data(), global_matrices_csr_data_d[index].data.data());
		//oneapi::mkl::sparse::optimize_gemv(q, oneapi::mkl::transpose::nontrans , jacobi_matrices[index].a_lu_handle);
		//oneapi::mkl::sparse::optimize_gemv(q, oneapi::mkl::transpose::nontrans , jacobi_matrices[index].a_d_handle);

		jacobi_matrices[index].size = non_bdry_mat_size;
	}

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	//Conducting testd for the lowest level matrix here 
	// std::int32_t mat_size = 81;
	// std::vector<std::int32_t> rows_lu;
	// std::vector<std::int32_t> cols_lu;
	// std::vector<float> vals_lu;
	// std::vector<std::int32_t> rows_d;
	// std::vector<std::int32_t> cols_d;
	// std::vector<float> vals_d;
	// globalstiffenssmatrix(mat_size , rows_lu , cols_lu, vals_lu , rows_d , cols_d,vals_d );
	// csr_data csr_mat_lu = coo_to_csr (49 , 49 , rows_lu.size(),rows_lu , cols_lu , vals_lu);
	// csr_data csr_mat_d = coo_to_csr (49,49,rows_d.size() , rows_d , cols_d , vals_d);
	// matrix_handle_t a_lu;
	// matrix_handle_t a_d;
	// init_matrix_handle (&a_lu);
	// init_matrix_handle (&a_d);
	// set_csr_data(a_lu, 49, 49, oneapi::mkl::index_base::zero, csr_mat_lu.indptr.data(), csr_mat_lu.indices.data(), csr_mat_lu.data.data());
	// set_csr_data(a_d, 49, 49, oneapi::mkl::index_base::zero, csr_mat_d.indptr.data(), csr_mat_d.indices.data(), csr_mat_d.data.data());
	// std::vector<float> test_vec(49 , 0.0);
	// std::vector<float> test_f(49 , 2.0);
	// // std::vector<float> temp_vec(49 , 0);
	// // std::vector<float> test_vec_2 (49 , 4.0);
	// std::vector<float> result = jacobirelaxation(a_lu , 49 , test_vec , test_f , 2);
	// for(auto e: result) {std::cout<<e<<" ";}
	// cl::sycl::queue q;
	// cl::sycl::event e1 = oneapi::mkl::vm::add(q, 49, test_vec.data(), test_f.data(), temp_vec.data(), {});
	// oneapi::mkl::vm::add(q, 49, temp_vec.data(), test_vec_2.data(), test_vec.data(), {e1}).wait();
	// //cl::sycl::event e1 = oneapi::mkl::blas::column_major::scal(q, test_f.size(), 2.0, test_f.data(), 1, {});
	// //oneapi::mkl::blas::column_major::scal(q, test_f.size(), 2.0, test_f.data(), 1, {e1}).wait();
	// for(auto e: test_vec) {std::cout<<e<<" ";}

	 ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	std::vector<float> f_global = globalforcefunction();
	//for(auto e: global_matrices_coo_data[0].rows_lu) {std::cout<<e<<" ";}
	std::vector<float> solution_finest = fullmultigrid(q, jacobi_matrices[jacobi_matrices.size() - 1], f_global);
	std::cout << "Size of finest level solution is " << solution_finest.size() << "\n";
	std::cout << "Program Running Correctly ";
	return 0;
}