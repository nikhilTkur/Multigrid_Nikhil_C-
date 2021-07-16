cmake_minimum_required(VERSION 3.12)

set(PROJECT_NAME demo_poisson_multigrid)
project(${PROJECT_NAME})


# Set C++ standard before finding pybind11
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)	
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "-Ofast -fsycl -DMKL_ILP64 -tbb ${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl-unnamed-lambda")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3 -DNDEBUG -fsycl -DMKL_ILP64 -tbb")


set(ONE_MKL_DIR "/opt/intel/oneapi/mkl/2021.2.0/")
include_directories($ENV{ONE_MKL_DIR})

# Get DOLFINx configuration data (DOLFINXConfig.cmake must be in
# DOLFINX_CMAKE_CONFIG_PATH)
if (NOT TARGET dolfinx)
  find_package(DOLFINX REQUIRED)
endif()
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
#include_directories(${Eigen3_DIR})
# Executable
add_executable(${PROJECT_NAME} main.cpp poisson.c Multigrid_functions.h)


target_link_libraries(${PROJECT_NAME} PUBLIC dolfinx "mkl_sycl;mkl_intel_ilp64;mkl_tbb_thread;mkl_core;sycl;OpenCL;tbb;pthread;dl;m")
# target_link_libraries(${PROJECT_NAME} dolfinx -fsycl -L${ONE_MKL_DIR}/lib/intel64 -lmkl_intel_ilp64 -lmkl_tbb_thread -lmkl_core -lpthread -lm -ldl)
# target_link_libraries(cpp PUBLIC m mkl_intel_lp64 mkl_core mkl_tbb_thread mkl_intel_thread iomp5 pthread)

# Do not throw error for 'multi-line comments' (these are typical in
# rst which includes LaTeX)
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-Wno-comment" HAVE_NO_MULTLINE)
set_source_files_properties(main.cpp PROPERTIES COMPILE_FLAGS "$<$<BOOL:${HAVE_NO_MULTLINE}>:-Wno-comment -Wextra -pedantic>")