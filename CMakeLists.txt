cmake_minimum_required(VERSION 3.18)

# Enable CUDA

project(cuda_challenge LANGUAGES CXX CUDA)
enable_language(CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Enable compile_commands.json generation for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS_JSON "${CMAKE_BINARY_DIR}/compile_commands.json")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA Toolkit version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Toolkit root: ${CUDAToolkit_ROOT}")
message(STATUS "CUDA Include directories: ${CUDAToolkit_INCLUDE_DIRS}")
# CCCL (Thrust, CUB, libcudacxx) are included in CUDA Toolkit
# Handle both CUDA 13.0+ (cccl/ subdirectory) and earlier versions (direct include)
# see: https://nvidia.github.io/cccl/cccl/3.0_migration_guide.html
if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13.0")
    # CUDA 13.0+: CCCL is in a separate cccl/ subdirectory
    set(CCCL_ROOT "${CUDAToolkit_INCLUDE_DIRS}/cccl")
    set(THRUST_INCLUDE_DIR "${CCCL_ROOT}/thrust")
    set(CUB_INCLUDE_DIR "${CCCL_ROOT}/cub")
    set(LIBCUDACXX_INCLUDE_DIR "${CCCL_ROOT}/cuda/include")
    message(STATUS "CUDA ${CUDAToolkit_VERSION}: Using CCCL from cccl/ subdirectory")
    add_compile_definitions(CCCL_VERSION_GREATER_EQUAL_13_0)
else()
    # CUDA < 13.0: CCCL components are directly in include directory
    set(THRUST_INCLUDE_DIR "${CUDAToolkit_INCLUDE_DIRS}/thrust")
    set(CUB_INCLUDE_DIR "${CUDAToolkit_INCLUDE_DIRS}/cub")
    set(LIBCUDACXX_INCLUDE_DIR "${CUDAToolkit_INCLUDE_DIRS}/cuda")
    message(STATUS "CUDA ${CUDAToolkit_VERSION}: Using CCCL from include/ directory")
endif()

# Set CUDA specific flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda --expt-relaxed-constexpr")

# Force CUDA compilation for specific files
set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/typedef.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vector_type_t.h
    PROPERTIES
    LANGUAGE CUDA
)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(AutoDetectCudaArch)
message(STATUS "CUDA_ARCHS: ${CUDA_ARCHS}")

# Extract compute capability from CUDA_ARCHS and set CMAKE_CUDA_ARCHITECTURES
if(CUDA_ARCHS)
    string(REGEX MATCH "compute_([0-9]+)" COMPUTE_CAPABILITY "${CUDA_ARCHS}")
    set(COMPUTE_CAPABILITY "${CMAKE_MATCH_1}")
    set(CMAKE_CUDA_ARCHITECTURES ${COMPUTE_CAPABILITY})
    message(STATUS "Setting CMAKE_CUDA_ARCHITECTURES to: ${CMAKE_CUDA_ARCHITECTURES}")
endif()
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
add_compile_definitions(CULBVH_ASSET_PATH="${CMAKE_CURRENT_SOURCE_DIR}/asset/")
find_package(TBB REQUIRED)
add_subdirectory(src)
add_subdirectory(tool)
option(ENABLE_TESTS "Enable tests" ON)
if(ENABLE_TESTS)
    message(STATUS "Building tests")
    add_subdirectory(test)
endif()
