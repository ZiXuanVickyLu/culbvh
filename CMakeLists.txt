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
