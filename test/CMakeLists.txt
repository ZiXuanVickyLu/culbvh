# Set CUDA compilation for test files
set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/test_random.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_simulation_dataset.cu
    PROPERTIES
    LANGUAGE CUDA
)

add_executable(lbvh_test test_random.cpp)
target_link_libraries(lbvh_test PRIVATE culbvh)

add_executable(lbvh_test_simulation_dataset test_simulation_dataset.cu)
target_link_libraries(lbvh_test_simulation_dataset PRIVATE culbvh culbvh_tools)

# Set CUDA properties for executables
set_target_properties(lbvh_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

set_target_properties(lbvh_test_simulation_dataset PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_include_directories(lbvh_test PRIVATE 
    ${CMAKE_SOURCE_DIR}/src
)
target_include_directories(lbvh_test_simulation_dataset PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tool
)


