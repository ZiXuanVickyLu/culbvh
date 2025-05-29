# Set CUDA compilation for all source files
set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/culbvh.cu
    PROPERTIES
    LANGUAGE CUDA
)

add_library(culbvh STATIC ${CMAKE_CURRENT_SOURCE_DIR}/culbvh.cu)

# Set CUDA specific properties for the libraries
set_target_properties(culbvh PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Link libraries and set include directories
target_link_libraries(culbvh PRIVATE TBB::tbb)

target_include_directories(culbvh PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR} 
    ${TBB_INCLUDE_DIRS}
)
