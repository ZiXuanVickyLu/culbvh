//
// BVH Visualization Application
// Reads config.json and generates wireframe OBJ files for BVH data
//

#include "aabb_wireframe.h"
#include "segment_mesh_writer.h"
#include "../src/culbvh.cuh"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>

using namespace culbvh;

#ifdef CURRENT_CONFIG_PATH
std::string get_current_config_path() {
    return std::string(CURRENT_CONFIG_PATH);
}
#else
std::string get_current_config_path() {
    return std::string("");
}
#endif

struct Config {
    std::string visualize_path;
    
    bool parse(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cout << "Error: Could not open config.json" << std::endl;
            return false;
        }
        
        std::string line;
        while (std::getline(file, line)) {
            // Remove whitespace and quotes
            line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());
            line.erase(std::remove(line.begin(), line.end(), '"'), line.end());
            line.erase(std::remove(line.begin(), line.end(), ','), line.end());
            
            if (line.find("visualize_path:") != std::string::npos) {
                visualize_path = line.substr(line.find(':') + 1);
            }
        }
        if(!visualize_path.empty()){
            visualize_path = get_asset_path() + visualize_path;
        }
        return !visualize_path.empty();
    }
    
    std::string getOutputName() const {
        // Extract filename without extension for output name
        size_t lastSlash = visualize_path.find_last_of("/\\");
        size_t lastDot = visualize_path.find_last_of(".");
        
        std::string filename = (lastSlash != std::string::npos) ? 
            visualize_path.substr(lastSlash + 1) : visualize_path;
            
        if (lastDot != std::string::npos && lastDot > lastSlash) {
            filename = filename.substr(0, lastDot - (lastSlash + 1));
        }
        
        return filename + "_bvh";
    }
};

// Load AABB data from binary file
std::vector<Bound<float>> loadAABBsFromBinary(const std::string& filename) {
    std::vector<Bound<float>> aabbs;
    std::ifstream file(filename, std::ios::binary);
    
    if (!file.is_open()) {
        std::cout << "Error: Could not open input file: " << filename << std::endl;
        return aabbs;
    }
    
    // Read file size to determine number of AABBs
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // Each AABB is stored as 6 floats (minx, miny, minz, maxx, maxy, maxz)
    size_t numAABBs = fileSize / (6 * sizeof(float));
    
    std::cout << "File size: " << fileSize << " bytes" << std::endl;
    std::cout << "Expected AABBs: " << numAABBs << std::endl;
    
    for (size_t i = 0; i < numAABBs; i++) {
        float data[6];
        file.read(reinterpret_cast<char*>(data), 6 * sizeof(float));
        
        if (file.gcount() == 6 * sizeof(float)) {
            auto min_point = make_float3(data[0], data[1], data[2]);
            auto max_point = make_float3(data[3], data[4], data[5]);
            aabbs.emplace_back(min_point, max_point);
        } else {
            break;
        }
    }
    
    std::cout << "Loaded " << aabbs.size() << " AABBs from " << filename << std::endl;
    return aabbs;
}

// Load AABB data from text file (fallback)
std::vector<Bound<float>> loadAABBsFromText(const std::string& filename) {
    std::vector<Bound<float>> aabbs;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cout << "Error: Could not open input file: " << filename << std::endl;
        return aabbs;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        float minx, miny, minz, maxx, maxy, maxz;
        
        if (iss >> minx >> miny >> minz >> maxx >> maxy >> maxz) {
            auto min_point = make_float3(minx, miny, minz);
            auto max_point = make_float3(maxx, maxy, maxz);
            aabbs.emplace_back(min_point, max_point);
        }
    }
    
    std::cout << "Loaded " << aabbs.size() << " AABBs from " << filename << std::endl;
    return aabbs;
}

int main() {
    std::cout << "CULBVH BVH Visualizer" << std::endl;
    std::cout << "=====================" << std::endl;
    
    // Parse config.json
    Config config;
    if (!config.parse(get_current_config_path() + "config.json")) {
        std::cout << "Error: Failed to parse config.json" << std::endl;
        std::cout << "Expected format:" << std::endl;
        std::cout << "{" << std::endl;
        std::cout << "    \"visualize_path\": \"path/to/aabb_data.bin\"" << std::endl;
        std::cout << "}" << std::endl;
        return -1;
    }
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Visualize path: " << config.visualize_path << std::endl;
    std::cout << "  Output name: " << config.getOutputName() << std::endl;
    std::cout << std::endl;
    
    // Load AABB data (try binary first, then text)
    std::vector<Bound<float>> host_aabbs;
    
    if (config.visualize_path.find(".bin") != std::string::npos) {
        host_aabbs = loadAABBsFromBinary(config.visualize_path);
    } else {
        host_aabbs = loadAABBsFromText(config.visualize_path);
    }
    
    if (host_aabbs.empty()) {
        std::cout << "Error: No AABB data loaded" << std::endl;
        return -1;
    }

    thrust::device_vector<Bound<float>> device_aabbs(host_aabbs);
    LBVH bvh;
    std::cout << "Building BVH..." << std::endl;
    bvh.compute(thrust::raw_pointer_cast(device_aabbs.data()), device_aabbs.size());
    bvh.bvhSelfCheck();
    
    std::cout << "BVH built successfully with " << device_aabbs.size() << " leaf nodes" << std::endl;

    AABBWireFrame<float> wireframe;
    std::cout << "Generating BVH wireframe..." << std::endl;
    wireframe.build(&bvh);
    
    if (!wireframe.is_valid()) {
        std::cout << "Error: Failed to generate BVH wireframe" << std::endl;
        return -1;
    }

    std::string output_path = get_asset_path() + "out/";

    SegmentMeshWriter<float> writer;
    writer.set_prefix(config.getOutputName());
    writer.set_path(output_path);
    writer.upload(wireframe.edges(), wireframe.points());
    
    std::cout << "Writing OBJ file..." << std::endl;
    writer.download();
    wireframe.destroy();
    
    std::cout << std::endl;
    std::cout << "BVH visualization completed successfully!" << std::endl;
    std::cout << "Output file: " << output_path << config.getOutputName() << "_0.obj" << std::endl;
    
    return 0;
}

