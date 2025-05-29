# Tool library for CULBVH visualization utilities

set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/aabb_wireframe.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/segment_mesh_writer.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/visualize_aabb.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/visualize_bvh.cpp
    PROPERTIES
    LANGUAGE CUDA
)

add_library(culbvh_tools STATIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/aabb_wireframe.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/segment_mesh_writer.cu
)

set_target_properties(culbvh_tools PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_link_libraries(culbvh_tools PUBLIC 
    culbvh
    TBB::tbb
)

target_include_directories(culbvh_tools PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    ${TBB_INCLUDE_DIRS}
)

add_executable(visualize_aabb 
    ${CMAKE_CURRENT_SOURCE_DIR}/visualize_aabb.cpp
)

add_executable(visualize_bvh 
    ${CMAKE_CURRENT_SOURCE_DIR}/visualize_bvh.cpp
)

set_target_properties(visualize_aabb PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

set_target_properties(visualize_bvh PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)


target_link_libraries(visualize_aabb PRIVATE 
    culbvh_tools
    TBB::tbb
)

target_link_libraries(visualize_bvh PRIVATE 
    culbvh_tools
    TBB::tbb
)
target_include_directories(visualize_aabb PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    ${TBB_INCLUDE_DIRS}
)

target_include_directories(visualize_bvh PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    ${TBB_INCLUDE_DIRS}
)

add_compile_definitions(CURRENT_CONFIG_PATH="${CMAKE_CURRENT_SOURCE_DIR}/")