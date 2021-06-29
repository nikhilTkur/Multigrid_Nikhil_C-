#include <CL/sycl.hpp>
#include<vector>
#include <unordered_map>

using namespace sycl;



std::vector<double> interpolation2D(std::vector<double>vec_2h, ProblemVar& obj, int target_level) {
	int vec_2h_dim = vec_2h.size();
	int vec_h_dim = int(std::pow(2 * (std::sqrt(vec_2h_dim) - 1) + 1, 2));
	std::vector<double> vec_h(vec_h_dim);
	for (int i = 0; i < vec_h_dim; i++) {
		int fine_space_dof = obj.topo_to_space_dict[target_level][i];
		if ((obj.parent_info_dict[target_level][i][0]) == 0) {
			// Coarse topo dof and fine topo dof coincide
			vec_h[fine_space_dof] = vec_2h[obj.topo_to_space_dict[target_level - 1][(obj.parent_info_dict[target_level][i][1])]];
		}
		else {
			//Fine dof lies on coarse edge
			int edge_num = (obj.parent_info_dict[target_level][i][1]);
			//Obtain corresponding edge vertices on topology
			int coarse_dof_1 = obj.topo_to_space_dict[target_level - 1][obj.coarse_grid_edges_dict[target_level - 1][edge_num][0]];
			int coarse_dof_2 = obj.topo_to_space_dict[target_level - 1][obj.coarse_grid_edges_dict[target_level - 1][edge_num][1]];
			vec_h[fine_space_dof] = 0.5 * (vec_2h[coarse_dof_1] + vec_2h[coarse_dof_2]);
		}
	}
	return vec_h;
}

//Define the restriction operator

std::vector<double> restriction2D(std::vector<double>vec_h, ProblemVar& obj, int target_level) {
	int vec_h_dim = vec_h.size();
	int vec_2h_dim = int(std::pow(((std::sqrt(vec_h_dim) - 1) / 2) + 1, 2));
	std::vector<double> vec_2h(vec_2h_dim);
	for (int i = 0; i < vec_2h_dim; i++) {
		vec_2h[obj.topo_to_space_dict[target_level][i]] = vec_h[obj.topo_to_space_dict[target_level + 1][i]];
	}
	return vec_2h;
}