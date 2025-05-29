//
// Created by birdpeople on 9/13/2023.
//

#ifndef SEGMENT_MESH_WRITER_H
#define SEGMENT_MESH_WRITER_H

#include "../src/typedef.h"
#include "../src/bound.h"
#include <memory>
#include <string>
#include <fstream>
#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
namespace culbvh {

    template<typename T = float>
    class SegmentMeshWriter {
    public:
        using point_type = typename Bound<T>::point_t;
        using edge_type = uint2;

        inline void download();
        inline void upload(std::shared_ptr<thrust::device_vector<edge_type>> edge_src,
                          std::shared_ptr<thrust::device_vector<point_type>> point_src) {
            m_edge = edge_src;
            m_point = point_src;
        }

        inline void set_prefix(std::string prefix) { m_prefix = std::move(prefix); }
        inline void set_path(std::string path) { m_path = std::move(path); }
        inline void reset() { m_frame = 0; }

    private:
        std::shared_ptr<thrust::device_vector<edge_type>> m_edge = nullptr;
        std::shared_ptr<thrust::device_vector<point_type>> m_point = nullptr;
        unsigned long long m_frame = 0;
        std::string m_prefix{"segment_mesh"};
        std::string m_path{get_asset_path() + "out/"};
    };

}

#endif //SEGMENT_MESH_WRITER_H
