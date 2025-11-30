#pragma once
// birdpeople, 2025, 11/28, implementation of stacklessbvh_lite.

#include <memory>
#include <string>

#include "typedef.h"
#include "vector_type_t.h"
#include "bound.h"
#ifndef CCCL_VERSION_GREATER_EQUAL_13_0
#include "thrust/device_vector.h"
#include "thrust/device_ptr.h"
#else
#include <cccl/thrust/device_vector.h>
#include <cccl/thrust/device_ptr.h>
#endif
#include "culbvh.cuh"

namespace culbvh {
	/// <summary>
	/// Very simple high-performance GPU StacklessLBVH that takes in a list of bounding boxes and outputs overlapping pairs.
	/// Side note: null bounds (inf, -inf) as inputs are ignored automatically.
	/// This implementation is not highly optimized, but serves as a reference for implementing the stackless traversal and build. I have add enough comment for understanding the algorithm.
	/// This implementation can fully pass the simulation dataset test.
	/// </summary>
	template<typename T>
	struct LBVHNode ;

	// 32-byte aligned node for stackless traversal
	struct __align__(32) StacklessNode {
		Bound<float> bound;
		int data;   // -1 for internal, primitive index for leaf
		int escape; // Index of next node to visit on miss/finish
	};

	class LBVHStacklessLite {
	public:
		using aabb = Bound<float>;
		using vec_type = float3;

		LBVHStacklessLite();
		~LBVHStacklessLite();

		// root bounds of every node in this tree, one large bv.
		aabb bounds() const;

		bool is_valid() const { return numObjs > 0; }

		size_t size() const { return numObjs; }

		const thrust::device_vector<LBVHNode<float>>& internal_nodes() const;

		const thrust::device_ptr<aabb>& object_aabbs() const;

		/// <summary>
		/// Refits an existing aabb tree once compute() has been called.
		/// Does not recompute the tree structure but only the AABBs.
		/// </summary>
		void refit();

		/// <summary>
		/// Allocates memory and builds the LBVH from a list of AABBs.
		/// Can be called multiple times for memory reuse.
		/// </summary>
		/// <param name="devicePtr">The device pointer containing the AABBs</param>
		/// <param name="size">The number of AABBs</param>
		void compute(aabb* devicePtr, size_t size);

		/// <summary>
		/// Tests this BVH against another BVH. Outputs unique collision pairs.
		/// The calling BVH should be the smaller one for best performance. 
		/// </summary>
		/// <param name="d_res">Device pointer with pairs containing (in order) the calling BVH object ID and then the other BVH object ID.</param>
		/// <param name="resSize">The number of entries allocated</param>
		/// <param name="d_otherAABBs">The other BVH AABBs</param>
		/// <param name="otherSize">The number of other BVH AABBs</param>
		/// <returns>The number of unique collision pairs written</returns>
		size_t query(int2* d_res, size_t resSize, aabb* d_otherAABBs, size_t otherSize) const;

		/// <summary>
		/// Tests this BVH against itself. Outputs unique collision pairs.
		/// </summary>
		/// <param name="d_res">Device pointer with unique object ID pairs</param>
		/// <param name="resSize">The number of entries allocated</param>
		/// <returns>The number of unique collision pairs written</returns>
		size_t query(int2* d_res, size_t resSize) const;

		/// <summary>
		/// Tests this BVH query using a ground truth method(cpu brute force). Outputs unique collision pairs.
		/// </summary>
		/// <param name="d_res">Device pointer with unique object ID pairs, should be filled with the result of query()!!!</param>
		/// <param name="resSize">The number of collision pairs</param>
		/// <returns>True if the query is correct, false otherwise</returns>
		bool query_compare_ground_truth(int2* d_res, size_t resSize) const;

		/// <summary>
		/// Tests this BVH query against another BVH using a ground truth method(cpu brute force). Outputs unique collision pairs.
		/// </summary>
		/// <param name="d_res">Device pointer with unique object ID pairs, should be filled with the result of query()!!!</param>
		/// <param name="resSize">The number of collision pairs</param>
		/// <param name="d_otherAABBs">The other BVH AABBs</param>
		/// <param name="otherSize">The number of other BVH AABBs</param>
		/// <returns>True if the query is correct, false otherwise</returns>
		bool query_compare_ground_truth(int2* d_res, size_t resSize, aabb* d_otherAABBs, size_t otherSize) const;

		// Does a self check of the BVH structure for debugging purposes.
		void bvhSelfCheck() const;

	private:
		struct thrustImpl;
		std::unique_ptr<thrustImpl> impl;
		aabb rootBounds;
		size_t numObjs {0};
	};

	// Tests the LBVH with a simple test case of 100k objects.
	void testLBVHStacklessLite();
}
