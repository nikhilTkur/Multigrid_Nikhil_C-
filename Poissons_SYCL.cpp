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


using namespace sycl;
using namespace oneapi::mkl::sparse;


// Defining the parameters of MultiGrid.
const int finest_level = 6;
const int coarsest_level = 3;
int highest_size = int(std::pow(2, finest_level));
const int mu0 = 30;
const int mu1 = 15;
const int mu2 = 15;

//Defining Global Matrices;
std::vector <std::pair<matrix_handle_t, int>> global_matrices(finest_level - coarsest_level + 1); // Matrices along with their sizes (sizes to be used for V Cycle and FMG Cycle)

//Define the Domain size of the problem
const float domain_x = 1.0;
const float domain_y = 1.0;

//Define the force vector of the Poissons Equation
const float f = 4.0;

std::vector<float> JacobiRelaxation(matrix_handle_t A, std::vector<float> &v, std::vector<float>&f, const int &mu) {
	
	return v;
}

//using namespace oneapi::mkl::sparse;
std::vector<std::vector<float>> triangle_element_stiffness_matrix(float &x1, float  &x2, float  &x3, float  &y1, float  &y2, float  &y3){
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

	std::vector <std::vector <float>> K_element(3, std::vector<float>(3,0.0));
	K_element[0][0] = (beta_1 * beta_1 + gamma_1 * gamma_1) / (4 * Ae);
	K_element[0][1] = (beta_1 * beta_2 + gamma_1 * gamma_2) / (4 * Ae);
	K_element[0][2] = (beta_1 * beta_3 + gamma_1 * gamma_3) / (4 * Ae);
	K_element[1][0] = (beta_2 * beta_1 + gamma_2 * gamma_1) / (4 * Ae);
	K_element[1][1] = (beta_2 * beta_2 + gamma_2 * gamma_2) / (4 * Ae);
	K_element[1][2] = (beta_2 * beta_3 + gamma_2 * gamma_3) / (4 * Ae);
	K_element[2][0] = (beta_3 * beta_1 + gamma_3 * gamma_1) / (4 * Ae);
	K_element[2][1] = (beta_3 * beta_2 + gamma_3 * gamma_2) / (4 * Ae);
	K_element[2][2] = (beta_3 * beta_3 + gamma_3 * gamma_3) / (4 * Ae);

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

void GlobalStiffenssMatrix(int &mat_size, matrix_handle_t handle) { // TODO
	int d_size = std::sqrt(mat_size) -1;
	float h = domain_x / d_size;
	std::vector <int> rows;
	std::vector <int> cols;
	std::vector <float> vals;
		



}
//TODO

std::vector <float> GlobalForceFunction() {
	//int sqrt_global_vector_size = int(std::pow(2, finest_level)) + 1;
	int global_vector_size = (highest_size + 1) * (highest_size + 1);
	float h = domain_x / (highest_size);
	std::vector<float> global_force_vector(global_vector_size , 0.0);
	int element = 1;
	for (int i = 1; i <= highest_size; i++) {

		for (int j = 1; j <= 2 * highest_size; j++) {

			if (element % 2 == 1) {
				float x1 = ((j - 1) * h) / 2;
				float x2 = x1;
				float x3 = ((j + 1) * h) / 2;
				float y1 = (i - 1) * h;
				float y2 = i * h;
				float y3 = y2;
				std::vector<float> fe = force_function_element(x1, x2, x3, y1, y2, y3);
				global_force_vector[(element + 1)/2 + i -1-1] += fe[0];
				global_force_vector[(element + 1) / 2 + i + highest_size - 1] += fe[1];
				global_force_vector[(element + 1) / 2 + highest_size + i + 1 - 1] += fe[2];
			}
			else {
				float x1 = (j * h) / 2;
				float x2 = x1;
				float x3 = ((j - 2) * h) / 2;
				float y1 = i * h;
				float y2 = (i - 1) * h;
				float y3 = y2;
				std::vector<float> fe = force_function_element(x1, x2, x3, y1, y2, y3);
				global_force_vector[element / 2 + i + highest_size + 1 - 1] += fe[0];
				global_force_vector[element / 2 + i - 1] += fe[1];
				global_force_vector[element / 2 + i + 1 - 1] += fe[2];
			}
			element = element + 1;

		}
	}


	// Extracting the non boundary elements from the Assembled Vector



	return global_force_vector;
		

}
//TODO

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

std::vector<float> VCycleMultigrid(std::pair<matrix_handle_t, int> &A_h, std::vector<float> &vec_h, std::vector<float> &f_h , sycl::queue &q) {

	matrix_handle_t A_handle = std::get<0>(A_h);
	int A_size = std::get<1>(A_h);
	std::vector<float> vec_2h;
	vec_h = JacobiRelaxation(A_handle, vec_h, f_h, mu1);

	if (int(std::log2(std::sqrt(A_size) + 1)) == coarsest_level) {

		vec_h = JacobiRelaxation(A_handle, vec_h, f_h, mu2);
		return vec_h;
	}
	else {
		//calculating the residual for the current solution
		std::vector<float> result(vec_h.size(), 0);
		std::vector<float> residual(f_h.size(), 0);
		std::vector<float> f_2h;
		const float a = 1.0;
		const float b = 0.0;
		const float* vec_ptr = vec_h.data();
		const float* result_ptr = result.data();
		cl::sycl::event gemv_done = cl::sycl::event();
		cl::sycl::event sub_done = cl::sycl::event();
		gemv_done = gemv(q, oneapi::mkl::transpose::nontrans, a, A_handle, vec_ptr, b, result_ptr, {});
		sub_done = oneapi::mkl::vm::sub(q, f_h.size(), f_h.data(), result.data(), residual.data(), { gemv_done });
		sub_done.wait();
		
		//Applying the restriction on the residual
		f_2h = Restriction2D(residual);
		vec_2h = std::vector<float> (f_2h.size(), 0);

		//Fetching the Coarser Level Matrix
		std::pair<matrix_handle_t, int> A_2h = global_matrices[int(std::log2(std::sqrt(f_2h.size()) + 1)) - coarsest_level];
		vec_2h = VCycleMultigrid(A_2h, vec_2h, f_2h, q);
	}
	std::vector<float> vec_2h_interpolation = Interpolation2D(vec_2h);
	cl::sycl::event vec_h_addition = cl::sycl::event();
	std::vector<float> vec_h_final(vec_h.size(), 0);
	vec_h_addition = oneapi::mkl::vm::add(q, vec_h.size(), vec_h.data(), vec_2h_interpolation.data(), vec_h_final.data(), {});
	vec_h_addition.wait();
	vec_h_final = JacobiRelaxation(A_handle, vec_h, f_h, mu2);
	return vec_h_final;
}

std::vector<float> FullMultiGrid(std::pair<matrix_handle_t, int>& A_h, std::vector<float>& f_h, sycl::queue& q) {
	std::vector<float> vec_h(f_h.size(), 0);
	std::vector<float> vec_2h;
	matrix_handle_t A_handle = std::get<0>(A_h);
	int A_size = std::get<1>(A_h);
	if (int(std::log2(std::sqrt(A_size) + 1)) == coarsest_level) {
		for (int i = 0; i <= mu0; i++) {
			vec_h = VCycleMultigrid(A_h, vec_h, f_h, q);
		}
		return vec_h;
	}
	else {
		std::vector<float> f_2h = Restriction2D(f_h);
		std::pair<matrix_handle_t, int> A_2h = global_matrices[int(std::log2(std::sqrt(f_2h.size()) + 1)) - coarsest_level];
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


}

	


