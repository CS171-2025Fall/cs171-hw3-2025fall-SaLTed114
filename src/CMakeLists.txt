find_package(OpenMP REQUIRED)

# ============================
# Enable CUDA
# ============================
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)

# Make sure CUDA targets build for RTX 4060 (SM 89)
set(CMAKE_CUDA_ARCHITECTURES 89)

# Include source files without main.cpp as lib
aux_source_directory("${PROJECT_SOURCE_DIR}/src" source)
message(STATUS "${source}")
list(REMOVE_ITEM source "${PROJECT_SOURCE_DIR}/src/main.cpp")
list(REMOVE_ITEM source "${PROJECT_SOURCE_DIR}/src/conventions.cpp")

# ============================
# Add GPU BVH CUDA sources
# ============================
file(GLOB GPU_CUDA_CU
    "${PROJECT_SOURCE_DIR}/src/gpu_bvh/*.cu"
)

# Merge into source list
list(APPEND source ${GPU_CUDA_CU})

# Set properties for CUDA files
set_source_files_properties(${GPU_BVH_CU} PROPERTIES LANGUAGE CUDA)

# # Add library
add_library(renderer_lib STATIC "${source}")
add_library(renderer::lib ALIAS renderer_lib)
set_property(TARGET renderer_lib PROPERTY CXX_STANDARD 17)

# Include libraries
target_link_libraries(renderer_lib PUBLIC 
  stb OpenMP::OpenMP_CXX fmt nlohmann_json tinyobjloader 
  linalg fmt spdlog tinyexr)
if (USE_EMBREE)
  target_link_libraries(renderer_lib PUBLIC embree)
  target_compile_definitions(renderer_lib PUBLIC USE_EMBREE)
  message(STATUS "Embree is enabled")
else()
  message(STATUS "Embree is disabled")
endif()

find_package(CUDAToolkit REQUIRED)
target_include_directories(renderer_lib PUBLIC
    "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_SOURCE_DIR}/src"
    ${CUDAToolkit_INCLUDE_DIRS}
)

add_executable(renderer "${PROJECT_SOURCE_DIR}/src/main.cpp")
target_link_libraries(renderer PRIVATE renderer::lib)

add_executable(conventions "${PROJECT_SOURCE_DIR}/src/conventions.cpp")
target_link_libraries(conventions PRIVATE renderer::lib)

add_executable(exrtools "${PROJECT_SOURCE_DIR}/src/exrtools.cpp")
target_link_libraries(exrtools PRIVATE renderer::lib)
