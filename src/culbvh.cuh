#pragma once
// Jerry Hsu, 2024

#include <memory>
#include <string>

#include "typedef.h"
#include "vector_type_t.h"
#include "bound.h"
#include "thrust/device_vector.h"
#include "thrust/device_ptr.h"

namespace culbvh {
	/// <summary>
	/// Very simple high-performance GPU LBVH that takes in a list of bounding boxes and outputs overlapping pairs.
	/// Side note: null bounds (inf, -inf) as inputs are ignored automatically.
	/// </summary>
	template<typename T>
	struct LBVHNode {
		Bound<T> bounds[2];  // Bounds for left and right children
		uint32_t leftIdx;    // Index of left child
		uint32_t rightIdx;   // Index of right child
		uint32_t parentIdx;  // Index of parent node
		uint32_t fence;      // Used for range queries
	};

	class LBVH {
	public:
		using aabb = Bound<float>;
		using vec_type = float3;

		// 64 byte node struct. Can fit two in a 128 byte cache line.
		struct alignas(64) node {
			uint32_t parentIdx;			// Parent node. Most siginificant bit (MSB) is used to indicate whether this is a left or right child of said parent.
			uint32_t leftIdx;			// Index of left child node. MSB is used to indicate whether this is a leaf node.
			uint32_t rightIdx;			// Index of right child node. MSB is used to indicate whether this is a leaf node.
			uint32_t fence;				// This subtree have indices between fence and current index.

			aabb bounds[2];
		};

		LBVH();
		~LBVH();

		// Returns the root bounds of every node in this tree.
		aabb bounds() const;

		// Returns whether the BVH is valid (has been built)
		bool is_valid() const { return numObjs > 0; }

		// Get the number of objects in the BVH
		size_t size() const { return numObjs; }

		// Get the internal nodes of the BVH
		const thrust::device_vector<LBVHNode<float>>& internal_nodes() const;

		// Get the object AABBs of the BVH
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
		/// <param name="other">The other BVH</param>
		/// <returns>The number of unique collision pairs written</returns>
		size_t query(int2* d_res, size_t resSize, LBVH* other) const;

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
		/// <param name="other">The other BVH</param>
		/// <returns>True if the query is correct, false otherwise</returns>
		bool query_compare_ground_truth(int2* d_res, size_t resSize, LBVH* other) const;

		// Does a self check of the BVH structure for debugging purposes.
		void bvhSelfCheck() const;

	private:
		struct thrustImpl;
		std::unique_ptr<thrustImpl> impl;
		aabb rootBounds;
		size_t numObjs {0};
		int maxStackSize;
	};

	// Tests the LBVH with a simple test case of 100k objects.
	void testLBVH();
}