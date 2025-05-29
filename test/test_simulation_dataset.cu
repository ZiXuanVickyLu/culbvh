//
// Created by bird_ on 5/23/2025.
//

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <set>
#include "aabb_wireframe.h"
#include "segment_mesh_writer.h"
// Platform-specific filesystem includes
#ifdef __has_include
#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#error "filesystem not available"
#endif
#else
#include <filesystem>
namespace fs = std::filesystem;
#endif

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "culbvh.cuh"
#include "bound.h"
#include "typedef.h"
using namespace culbvh;
std::vector<culbvh::Bound<float>> load_aabb_file(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filepath << std::endl;
        return {};
    }

    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    size_t num_aabbs = file_size / (6 * sizeof(float));
    
    std::vector<culbvh::Bound<float>> aabbs;
    aabbs.reserve(num_aabbs);
    
    for (size_t i = 0; i < num_aabbs; i++) {
        float data[6];
        file.read(reinterpret_cast<char*>(data), 6 * sizeof(float));
        
        if (file.gcount() != 6 * sizeof(float)) {
            std::cerr << "Error: Incomplete read from file " << filepath << std::endl;
            break;
        }
        
        float3 min_pt = make_float3(data[0], data[1], data[2]);
        float3 max_pt = make_float3(data[3], data[4], data[5]);
        aabbs.emplace_back(min_pt, max_pt);
    }
    
    file.close();
    return aabbs;
}

std::vector<int> get_frame_numbers(const std::string& dataset_path) {
    std::vector<int> frame_numbers;
    
    try {
        for (const auto& entry : fs::directory_iterator(dataset_path)) {
            if (entry.is_regular_file()) {
                std::string filename = entry.path().filename().string();
                if (filename.find("mesh_triangle_aabb_frame_") == 0) {
                    // Extract frame number from filename
                    size_t start = filename.find("_frame_") + 7;
                    size_t end = filename.find(".bin");
                    if (end != std::string::npos) {
                        std::string frame_str = filename.substr(start, end - start);
                        int frame_num = std::stoi(frame_str);
                        frame_numbers.push_back(frame_num);
                    }
                }
            }
        }
    } catch (const fs::filesystem_error& ex) {
        std::cerr << "Error accessing directory " << dataset_path << ": " << ex.what() << std::endl;
    }
    
    std::sort(frame_numbers.begin(), frame_numbers.end());
    return frame_numbers;
}

struct TestResult {
    int frame;
    float time;
    size_t mesh_count;
    size_t collider_count;
    float collider_build_time;
    float mesh_build_time;
    float query_time;
    float self_query_time;
    size_t query_contacts;
    size_t self_query_contacts;
    bool query_matches_ground_truth;
    bool self_query_matches_ground_truth;
};

void save_results_to_csv(const std::string& dataset_name, const std::vector<TestResult>& results) {
    std::string output_dir = culbvh::get_asset_path() + "statistic/";
    
    fs::create_directories(output_dir);
    
    std::string csv_path = output_dir + dataset_name + ".csv";
    std::ofstream csv_file(csv_path);
    
    if (!csv_file.is_open()) {
        std::cerr << "Error: Cannot create CSV file " << csv_path << std::endl;
        return;
    }
    
    csv_file << "frame,time,mesh_count,collider_count,"
             << "collider_build_time,mesh_build_time,query_time,self_query_time,"
             << "query_contacts,self_query_contacts,"
             << "query_matches_ground_truth,self_query_matches_ground_truth\n";
    
    for (const auto& result : results) {
        csv_file << result.frame << "," 
                << std::fixed << std::setprecision(2) << result.time << ","
                << result.mesh_count << "," << result.collider_count << ","
                << std::fixed << std::setprecision(3) 
                << result.collider_build_time << "," << result.mesh_build_time << ","
                << result.query_time << "," << result.self_query_time << ","
                << result.query_contacts << "," << result.self_query_contacts << ","
                << (result.query_matches_ground_truth ? "true" : "false") << ","
                << (result.self_query_matches_ground_truth ? "true" : "false") << "\n";
    }
    
    csv_file.close();
    std::cout << "Results saved to " << csv_path << std::endl;
}

TestResult test_frame(const std::string& dataset_name, int frame_num) {
    TestResult result;
    result.frame = frame_num;
    result.time = frame_num / 50.0f;  // Convert frame number to time
    result.query_matches_ground_truth = true;
    result.self_query_matches_ground_truth = true;
    
    std::string asset_path = culbvh::get_asset_path();
    std::string mesh_file = asset_path + dataset_name + "/mesh_triangle_aabb_frame_" + 
                           std::string(6 - std::to_string(frame_num).length(), '0') + std::to_string(frame_num) + ".bin";
    std::string collider_file = asset_path + dataset_name + "/moving_collider_triangle_aabb_frame_" + 
                               std::string(6 - std::to_string(frame_num).length(), '0') + std::to_string(frame_num) + ".bin";
    
    std::cout << "Processing frame " << frame_num << " (t=" << result.time << "s)" << std::endl;
    auto mesh_aabbs = load_aabb_file(mesh_file);
    auto collider_aabbs = load_aabb_file(collider_file);
    
    if (mesh_aabbs.empty() || collider_aabbs.empty()) {
        std::cerr << "Error: Failed to load AABB files for frame " << frame_num << std::endl;
        result.query_matches_ground_truth = false;
        result.self_query_matches_ground_truth = false;
        return result;
    }
    
    result.mesh_count = mesh_aabbs.size();
    result.collider_count = collider_aabbs.size();
    
    std::cout << "  Mesh AABBs: " << mesh_aabbs.size() << ", Collider AABBs: " << collider_aabbs.size() << std::endl;
    thrust::device_vector<culbvh::Bound<float>> d_mesh_aabbs = mesh_aabbs;
    thrust::device_vector<culbvh::Bound<float>> d_collider_aabbs = collider_aabbs;
    
    // Test culbvh with ground truth comparison
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        std::cout << "  Testing culbvh..." << std::endl;
        
        // Build collider BVH
        cudaEventRecord(start);
        culbvh::LBVH collider_bvh;
        collider_bvh.compute(thrust::raw_pointer_cast(d_collider_aabbs.data()), collider_aabbs.size());
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&result.collider_build_time, start, stop);
        collider_bvh.bvhSelfCheck();
        // Build mesh BVH
        cudaEventRecord(start);
        culbvh::LBVH mesh_bvh;
        mesh_bvh.compute(thrust::raw_pointer_cast(d_mesh_aabbs.data()), mesh_aabbs.size());
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&result.mesh_build_time, start, stop);
        mesh_bvh.bvhSelfCheck();
        // Query between collider and mesh
        size_t max_results = mesh_aabbs.size() * 128;  // Maximum possible collisions
        thrust::device_vector<int2> d_results(max_results);
        
        cudaEventRecord(start);
        result.query_contacts = mesh_bvh.query(thrust::raw_pointer_cast(d_results.data()), max_results, &collider_bvh);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&result.query_time, start, stop);
        
        // Compare with ground truth for collider-mesh query
        result.query_matches_ground_truth = mesh_bvh.query_compare_ground_truth(
            thrust::raw_pointer_cast(d_results.data()), result.query_contacts, &collider_bvh);
        
        // Self-query for mesh BVH
        auto self_max_results = mesh_bvh.size() * 128;
        thrust::device_vector<int2> d_self_results(self_max_results);
        
        cudaEventRecord(start);
        result.self_query_contacts = mesh_bvh.query(thrust::raw_pointer_cast(d_self_results.data()), self_max_results);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&result.self_query_time, start, stop);
        
        // Compare with ground truth for mesh self-query
        result.self_query_matches_ground_truth = mesh_bvh.query_compare_ground_truth(
            thrust::raw_pointer_cast(d_self_results.data()), result.self_query_contacts);
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        
        std::cout << "    Collider Build: " << result.collider_build_time << "ms" << std::endl;
        std::cout << "    Mesh Build: " << result.mesh_build_time << "ms" << std::endl;
        std::cout << "    Query: " << result.query_time << "ms, Contacts: " << result.query_contacts 
                 << ", Matches Ground Truth: " << (result.query_matches_ground_truth ? "Yes" : "No") << std::endl;
        std::cout << "    Self-Query: " << result.self_query_time << "ms, Contacts: " << result.self_query_contacts 
                 << ", Matches Ground Truth: " << (result.self_query_matches_ground_truth ? "Yes" : "No") << std::endl;
    }
    
    return result;
}

void test_dataset(const std::string& dataset_name) {
    std::cout << "\n=== Testing dataset: " << dataset_name << " ===" << std::endl;
    
    std::string dataset_path = culbvh::get_asset_path() + dataset_name + "/";
    auto frame_numbers = get_frame_numbers(dataset_path);
    
    if (frame_numbers.empty()) {
        std::cerr << "Error: No frames found for dataset " << dataset_name << std::endl;
        return;
    }
    
    std::cout << "Found " << frame_numbers.size() << " frames" << std::endl;
    
    std::vector<TestResult> results;
    
    size_t max_frames = std::min(frame_numbers.size(), size_t(100));
    for (size_t i = 0; i < max_frames; i++) {
        int frame_num = frame_numbers[i];
        auto result = test_frame(dataset_name, frame_num);
        results.push_back(result);
    }
    
    save_results_to_csv(dataset_name, results);
    
    std::cout << "\n=== Summary for " << dataset_name << " ===" << std::endl;
    float total_collider_build = 0, total_mesh_build = 0;
    float total_query = 0, total_self_query = 0;
    int query_errors = 0, self_query_errors = 0;
    
    for (const auto& result : results) {
        total_collider_build += result.collider_build_time;
        total_mesh_build += result.mesh_build_time;
        total_query += result.query_time;
        total_self_query += result.self_query_time;
        if (!result.query_matches_ground_truth) query_errors++;
        if (!result.self_query_matches_ground_truth) self_query_errors++;
    }
    
    std::cout << "Average build time - Collider: " << (total_collider_build / results.size()) 
             << "ms, Mesh: " << (total_mesh_build / results.size()) << "ms" << std::endl;
    std::cout << "Average query time - Cross-BVH: " << (total_query / results.size()) 
             << "ms, Self-query: " << (total_self_query / results.size()) << "ms" << std::endl;
    std::cout << "Errors - Cross-BVH: " << query_errors << "/" << results.size() 
             << ", Self-query: " << self_query_errors << "/" << results.size() << std::endl;
}

int main() {
    std::cout << "CUDA BVH Simulation Dataset Test" << std::endl;
    std::cout << "=================================" << std::endl;
    
    cudaSetDevice(0);
    
    std::string asset_path = culbvh::get_asset_path();
    std::cout << "Asset path: " << asset_path << std::endl;
    if (!fs::exists(asset_path + "dance/")) {
        std::cerr << "Error: Dance dataset not found at " << asset_path + "dance/" << std::endl;
        return 1;
    }
    
    if (!fs::exists(asset_path + "multilayer/")) {
        std::cerr << "Error: Multilayer dataset not found at " << asset_path + "multilayer/" << std::endl;
        return 1;
    }
    
    test_dataset("dance");
    test_dataset("multilayer");
    
    std::cout << "\nAll tests completed!" << std::endl;
    
    return 0;
}
