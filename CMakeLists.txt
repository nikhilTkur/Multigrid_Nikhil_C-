cmake_minimum_required(VERSION 3.12)

set(PROJECT_NAME demo_poisson_multigrid)
project(${PROJECT_NAME})


# Set C++ standard before finding pybind11
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)	
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "-Ofast -fsycl -DMKL_ILP64 -tbb ${CMAKE_CXX_FLAGS} -Wall")

include_directories($ENV{ONE_MKL_DIR})

# Get DOLFINx configuration data (DOLFINXConfig.cmake must be in
# DOLFINX_CMAKE_CONFIG_PATH)
if (NOT TARGET dolfinx)
  find_package(DOLFINX REQUIRED)
endif()
find_package(BLAS REQUIRED)

# Executable
add_executable(${PROJECT_NAME} main.cpp poisson.c Multigrid_functions.h)

target_link_libraries(${PROJECT_NAME} dolfinx)
target_link_libraries(${PROJECT_NAME} PRIVATE -fsycl)
target_link_libraries(cpp PUBLIC m mkl_intel_lp64 mkl_core mkl_tbb_thread mkl_intel_thread iomp5 pthread)

# Do not throw error for 'multi-line comments' (these are typical in
# rst which includes LaTeX)
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-Wno-comment" HAVE_NO_MULTLINE)
set_source_files_properties(main.cpp PROPERTIES COMPILE_FLAGS "$<$<BOOL:${HAVE_NO_MULTLINE}>:-Wno-comment -Wall -Wextra -pedantic -Werror>")