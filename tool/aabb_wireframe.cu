//
// Created by birdpeople on 12/4/2023.
//

#include "aabb_wireframe.h"
#include <thrust/count.h>
#include <thrust/copy.h>
#include <cuda_runtime.h>
#include "../src/typedef.h"

namespace culbvh {

    template<typename T>
    struct aabb_valid_predicate {
        CUDA_INLINE_CALLABLE bool operator()(const Bound<T>& aabb) const {
            return aabb.min.x <= aabb.max.x && 
                   aabb.min.y <= aabb.max.y && 
                   aabb.min.z <= aabb.max.z &&
                   aabb.min.x != std::numeric_limits<T>::max() &&
                   aabb.max.x != std::numeric_limits<T>::lowest();
        }
    };

    template<typename T>
    __global__ void create_wireframe_kernel(
        const Bound<T>* aabb_list,
        typename Bound<T>::point_t* points,
        uint2* edges,
        int num_aabbs
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_aabbs) return;

        const auto& aabb = aabb_list[idx];
        
        // Create 8 corner points for each AABB
        int point_base = idx * 8;
        points[point_base + 0] = make_point<T>(aabb.min.x, aabb.min.y, aabb.min.z);
        points[point_base + 1] = make_point<T>(aabb.max.x, aabb.min.y, aabb.min.z);
        points[point_base + 2] = make_point<T>(aabb.min.x, aabb.max.y, aabb.min.z);
        points[point_base + 3] = make_point<T>(aabb.max.x, aabb.max.y, aabb.min.z);
        points[point_base + 4] = make_point<T>(aabb.min.x, aabb.min.y, aabb.max.z);
        points[point_base + 5] = make_point<T>(aabb.max.x, aabb.min.y, aabb.max.z);
        points[point_base + 6] = make_point<T>(aabb.min.x, aabb.max.y, aabb.max.z);
        points[point_base + 7] = make_point<T>(aabb.max.x, aabb.max.y, aabb.max.z);

        // Create 12 edges for each AABB (wireframe cube)
        int edge_base = idx * 12;
        // Bottom face edges
        edges[edge_base + 0] = make_uint2(point_base + 0, point_base + 1);
        edges[edge_base + 1] = make_uint2(point_base + 1, point_base + 3);
        edges[edge_base + 2] = make_uint2(point_base + 3, point_base + 2);
        edges[edge_base + 3] = make_uint2(point_base + 2, point_base + 0);
        
        // Top face edges
        edges[edge_base + 4] = make_uint2(point_base + 4, point_base + 5);
        edges[edge_base + 5] = make_uint2(point_base + 5, point_base + 7);
        edges[edge_base + 6] = make_uint2(point_base + 7, point_base + 6);
        edges[edge_base + 7] = make_uint2(point_base + 6, point_base + 4);
        
        // Vertical edges
        edges[edge_base + 8] = make_uint2(point_base + 0, point_base + 4);
        edges[edge_base + 9] = make_uint2(point_base + 1, point_base + 5);
        edges[edge_base + 10] = make_uint2(point_base + 2, point_base + 6);
        edges[edge_base + 11] = make_uint2(point_base + 3, point_base + 7);
    }

    template<typename T>
    void AABBWireFrame<T>::build(const thrust::device_vector<aabb>& d_aabb) {
        if (d_aabb.empty()) return;

        // Count valid AABBs
        auto valid_count = thrust::count_if(thrust::device, d_aabb.begin(), d_aabb.end(), 
                                          aabb_valid_predicate<T>());
        printf("valid_count: %d\n", valid_count);
        if (valid_count == 0) return;
        if(valid_count!=d_aabb.size()){
            printf("Invalid AABBs found\n");
        }
        // Filter valid AABBs
        thrust::device_vector<aabb> d_valid_aabb(valid_count);
        thrust::copy_if(thrust::device, d_aabb.begin(), d_aabb.end(), 
                       d_valid_aabb.begin(), aabb_valid_predicate<T>());

        // Clean up previous data
        destroy();

        // Allocate memory for points and edges
        m_config.d_point = std::make_shared<thrust::device_vector<point>>(valid_count * 8);
        m_config.d_edge = std::make_shared<thrust::device_vector<edge>>(valid_count * 12);
        m_config.h_edge_index = std::make_shared<thrust::host_vector<index>>(valid_count * 24);
        m_config.bound_list_cnt = std::make_shared<unsigned int>(valid_count);

        // Launch kernel to create wireframe
        dim3 block_size(256);
        dim3 grid_size((valid_count + block_size.x - 1) / block_size.x);
        
        create_wireframe_kernel<T><<<grid_size, block_size>>>(
            thrust::raw_pointer_cast(d_valid_aabb.data()),
            thrust::raw_pointer_cast(m_config.d_point->data()),
            thrust::raw_pointer_cast(m_config.d_edge->data()),
            valid_count
        );

        // Copy edge indices to host
        thrust::copy(m_config.d_edge->begin(), m_config.d_edge->end(), 
                    reinterpret_cast<uint2*>(m_config.h_edge_index->data()));

        cudaDeviceSynchronize();
        m_is_valid = true;
    }

    template<typename T>
    void AABBWireFrame<T>::build(const LBVH* bvh) {
        if (!bvh->is_valid()) return;
        
        // Calculate total number of AABBs (internal nodes * 2 + object nodes)
        size_t num_internal_nodes = bvh->internal_nodes().size();
        size_t num_objects = bvh->size();
        size_t total_aabbs = num_internal_nodes + num_objects;
        
        // Create device vector for all AABBs
        thrust::device_vector<aabb> d_aabb(total_aabbs);
        printf("total_aabbs: %d\n", total_aabbs);
        printf("num_objects: %d\n", num_objects);

        // Copy object AABBs
        thrust::copy(thrust::device, 
                    bvh->object_aabbs(), 
                    bvh->object_aabbs() + num_objects, 
                    d_aabb.begin());
        
        // Create a kernel to copy internal node bounds
        auto d_nodes = bvh->internal_nodes();
        auto d_nodes_ptr = thrust::raw_pointer_cast(d_nodes.data());
        auto d_aabb_ptr = thrust::raw_pointer_cast(d_aabb.data() + num_objects);
        
        dim3 block_size(256);
        dim3 grid_size((num_internal_nodes + block_size.x - 1) / block_size.x);
        
        // Launch kernel to copy internal node bounds
        create_internal_bounds_kernel<<<grid_size, block_size>>>(
            d_nodes_ptr, d_aabb_ptr, num_internal_nodes);
        
        // Build the wireframe with all AABBs
        build(d_aabb);
    }

    template<typename T>
    __global__ void create_internal_bounds_kernel(
        const LBVHNode<T>* nodes,
        Bound<T>* aabbs,
        int num_nodes
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_nodes) return;
        if(!aabb_valid_predicate<T>()(nodes[idx].bounds[0]) || !aabb_valid_predicate<T>()(nodes[idx].bounds[1])){
            printf("Invalid AABBs found at%d: %f,%f,%f,%f,%f,%f\n", idx, nodes[idx].bounds[0].min.x, nodes[idx].bounds[0].min.y, nodes[idx].bounds[0].min.z, nodes[idx].bounds[0].max.x, nodes[idx].bounds[0].max.y, nodes[idx].bounds[0].max.z);
        }
        // Create a new AABB that encompasses both bounds
        Bound<T> merged_bounds = nodes[idx].bounds[0];
        merged_bounds.absorb(nodes[idx].bounds[1]);
        aabbs[idx] = merged_bounds;
       // printf("merged_bounds: %f,%f,%f,%f,%f,%f\n", merged_bounds.min.x, merged_bounds.min.y, merged_bounds.min.z, merged_bounds.max.x, merged_bounds.max.y, merged_bounds.max.z);
    }

    template<typename T>
    void AABBWireFrame<T>::destroy() {
        m_config.d_point.reset();
        m_config.d_edge.reset();
        m_config.h_edge_index.reset();
        m_config.bound_list_cnt.reset();
        m_is_valid = false;
    }

    // Explicit template instantiations
    template class AABBWireFrame<float>;
    // temporary disable double instantiation

} // namespace culbvh 