//
// Created by birdpeople on 12/4/2023.
//

#ifndef AABB_WIREFRAME_H
#define AABB_WIREFRAME_H

#include "../src/typedef.h"
#include "../src/bound.h"
#include "../src/culbvh.cuh"
#include <memory>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
namespace culbvh {
    template<typename T = float>
    struct AABBWireFrameConfig {
        std::shared_ptr<thrust::device_vector<typename Bound<T>::point_t>> d_point = nullptr;
        std::shared_ptr<thrust::device_vector<uint2>> d_edge = nullptr;  
        std::shared_ptr<thrust::host_vector<unsigned int>> h_edge_index = nullptr;
        std::shared_ptr<unsigned int> bound_list_cnt = nullptr;
    };

    template<typename T = float>
    struct AABBWireFrame{
        using point = typename Bound<T>::point_t;
        using edge = uint2;
        using index = unsigned int;
        using aabb = Bound<T>;
        
        void build(const thrust::device_vector<aabb>& d_aabb);
        void build(const LBVH* bvh);
        
        [[nodiscard]] bool is_valid() const { return m_is_valid;}
        [[nodiscard]] const AABBWireFrameConfig<T>& config() const { return m_config;}
        [[nodiscard]] AABBWireFrameConfig<T>& config() { return m_config;}
        [[nodiscard]] std::shared_ptr<thrust::device_vector<point>>& points() { return m_config.d_point;}
        [[nodiscard]] const std::shared_ptr<thrust::device_vector<point>>& points() const { return m_config.d_point;}
        [[nodiscard]] std::shared_ptr<thrust::device_vector<edge>>& edges() { return m_config.d_edge;}
        [[nodiscard]] const std::shared_ptr<thrust::device_vector<edge>>& edges() const { return m_config.d_edge;}
        void destroy();
        
    private:
        bool m_is_valid = false;
        AABBWireFrameConfig<T> m_config{};
    };
}

#endif //AABB_WIREFRAME_H
