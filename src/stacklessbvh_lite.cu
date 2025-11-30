#include "stacklessbvh_lite.cuh"

#ifndef CCCL_VERSION_GREATER_EQUAL_13_0
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/swap.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#else
#include <cccl/thrust/host_vector.h>
#include <cccl/thrust/device_vector.h>
#include <cccl/thrust/swap.h>
#include <cccl/thrust/host_vector.h>
#include <cccl/thrust/device_vector.h>
#include <cccl/thrust/functional.h>
#include <cccl/thrust/sort.h>
#include <cccl/thrust/fill.h>
#include <cccl/thrust/reduce.h>
#include <cccl/thrust/execution_policy.h>
#endif

#include <unordered_set>
#include <tbb/parallel_for.h>
#include <atomic>
#include <tbb/concurrent_unordered_set.h>
#include <chrono>

namespace culbvh {

	template<typename T>
    struct culbvh_aabb_valid_predicate {
        CUDA_INLINE_CALLABLE bool operator()(const Bound<T>& aabb) const {
            return aabb.min.x <= aabb.max.x && 
                   aabb.min.y <= aabb.max.y && 
                   aabb.min.z <= aabb.max.z &&
                   aabb.min.x != std::numeric_limits<T>::max() &&
                   aabb.max.x != std::numeric_limits<T>::lowest();
        }
    };

	// Custom hash function for int2
	struct Int2Hash {
		std::size_t operator()(const int2& k) const {
			// Simple hash function combining x and y
			return std::hash<int>()(k.x) ^ (std::hash<int>()(k.y) << 1);
		}
	};

	// Custom equality function for int2
	struct Int2Equal {
		bool operator()(const int2& lhs, const int2& rhs) const {
			return lhs.x == rhs.x && lhs.y == rhs.y;
		}
	};

	struct LBVHStacklessLite::thrustImpl {
		thrust::device_ptr<Bound<float>> d_objs = nullptr;
		thrust::device_vector<int> d_flags;				// Flags used for updating the tree

		thrust::device_vector<uint32_t> d_morton;		// Morton codes for each object
		thrust::device_vector<uint32_t> d_objIDs;		// Object ID for each leaf
		thrust::device_vector<uint32_t> d_leafParents;	// Parent ID for each leaf. MSB is used to indicate whether this is a left or right child of said parent.
		thrust::device_vector<LBVHNode<float>> d_nodes;	// The internal tree nodes
		
		// Stackless Data
		thrust::device_vector<StacklessNode> d_stackless_nodes;
		thrust::device_vector<int> d_subtree_sizes;
		
		// Pre-order layout indices (computed in parallel, no reorder needed)
		thrust::device_vector<int> d_internalPreorder;   // Preorder index for each internal node
		thrust::device_vector<int> d_internalEscape;     // Escape parent for each internal node
		thrust::device_vector<int> d_leafPreorder;       // Preorder index for each leaf
		thrust::device_vector<int> d_leafEscape;         // Escape parent for each leaf
	};

#pragma region LBVHStacklessLiteDevice

	namespace LBVHStacklessLiteKernels {
		// Custom comparison for int3 based on lexicographical ordering
		__device__ inline bool lessThan(const int3& a, const int3& b) {
			if (a.x != b.x) return a.x < b.x;
			if (a.y != b.y) return a.y < b.y;
			return a.z < b.z;
		}

		// We do 128 wide blocks which gets us 8KB of shared memory per block on compute 8.6. 
		// Ideally leave some left for L1 cache.
		__device__ constexpr int MAX_RES_PER_BLOCK = 4 * 128;

		// Computes the morton codes for each AABB
		template<typename T>
		__global__ void mortonKernel(Bound<T>* aabbs, uint32_t* codes, uint32_t* ids, Bound<T> wholeAABB, int size) {
			const int tid = threadIdx.x + blockIdx.x * blockDim.x;
			if (tid >= size) return;
			typename Bound<T>::point_t coord = wholeAABB.normCoord(aabbs[tid].center()) * 1024.f;
			int3 point = make_int3(coord.x, coord.y, coord.z);
			int3 min_point = make_int3(0, 0, 0);
			int3 max_point = make_int3(1023, 1023, 1023);
			
			// Manual clamping
			point.x = max(min_point.x, min(point.x, max_point.x));
			point.y = max(min_point.y, min(point.y, max_point.y));
			point.z = max(min_point.z, min(point.z, max_point.z));
			
			codes[tid] = culbvh::getMorton(point);
			ids[tid] = tid;
		}

		//counting leading zeros
		__device__ inline int commonUpperBits(const uint64_t lhs, const uint64_t rhs) {
			return ::__clzll(lhs ^ rhs);
		}

		// Merges morton code with its index to output a sorted unique 64-bit key.
		__device__ inline uint64_t mergeIdx(const uint32_t code, const int idx) {
			return ((uint64_t)code << 32ul) | (uint64_t)idx;
		}

		// see reference on Karras, 2012 "Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees"
		__device__ inline int2 determineRange(uint32_t const* mortonCodes,
			const uint32_t numObjs, uint32_t idx) {

			if (idx == 0)
				return make_int2(0, numObjs - 1);

			// Determine direction of the range
			const uint64_t selfCode = mergeIdx(mortonCodes[idx], idx);
			const int lDelta = commonUpperBits(selfCode, mergeIdx(mortonCodes[idx - 1], idx - 1));
			const int rDelta = commonUpperBits(selfCode, mergeIdx(mortonCodes[idx + 1], idx + 1));
			const int d = (rDelta > lDelta) ? 1 : -1;

			// Compute upper bound for the length of the range
			const int minDelta = thrust::min(lDelta, rDelta);
			int lMax = 2;
			int i;
			while ((i = idx + d * lMax) >= 0 && i < numObjs) {
				if (commonUpperBits(selfCode, mergeIdx(mortonCodes[i], i)) <= minDelta) break;
				lMax <<= 1;
			}

			// Find the exact range by binary search
			int t = lMax >> 1;
			int l = 0;
			while (t > 0) {
				i = idx + (l + t) * d;
				if (0 <= i && i < numObjs)
					if (commonUpperBits(selfCode, mergeIdx(mortonCodes[i], i)) > minDelta)
						l += t;
				t >>= 1;
			}

			unsigned int jdx = idx + l * d;
			if (d < 0) thrust::swap(idx, jdx); // Make sure that idx < jdx
			return make_int2(idx, jdx);
		}

		__device__ inline uint32_t findSplit(uint32_t const* mortonCodes,
			const uint32_t first, const uint32_t last) {

			const uint64_t firstCode = mergeIdx(mortonCodes[first], first);
			const uint64_t lastCode = mergeIdx(mortonCodes[last], last);
			const int deltaNode = commonUpperBits(firstCode, lastCode);

			// Binary search for split position
			int split = first;
			int stride = last - first;
			do {
				stride = (stride + 1) >> 1;
				const int middle = split + stride;
				if (middle < last)
					if (commonUpperBits(firstCode, mergeIdx(mortonCodes[middle], middle)) > deltaNode)
						split = middle;
			} while (stride > 1);

			return split;
		}

		// Builds out the internal nodes of the LBVH
		__global__ void lbvhBuildInternalKernel(LBVHNode<float>* nodes,
			uint32_t* leafParents, uint32_t const* mortonCodes, uint32_t const* objIDs, int numObjs) {

			const int tid = threadIdx.x + blockIdx.x * blockDim.x;
			if (tid >= numObjs - 1) return;

			int2 range = determineRange(mortonCodes, numObjs, tid);
			nodes[tid].fence = (tid == range.x) ? range.y : range.x;

			const int gamma = findSplit(mortonCodes, range.x, range.y);

			if (range.x == gamma) {
				leafParents[gamma] = (uint32_t)tid;
				range.x = gamma | 0x80000000;
			}
			else {
				range.x = gamma;
				nodes[range.x].parentIdx = (uint32_t)tid;
			}

			if (range.y == gamma + 1) {
				leafParents[gamma + 1] = (uint32_t)tid | 0x80000000;
				range.y = (gamma + 1) | 0x80000000;
			}
			else {
				range.y = gamma + 1;
				nodes[range.y].parentIdx = (uint32_t)tid | 0x80000000;
			}

			nodes[tid].leftIdx = range.x;
			nodes[tid].rightIdx = range.y;
		}

		// Refits the AABBs of the internal nodes
		__global__ void mergeUpKernel(LBVHNode<float>* nodes,
			uint32_t* leafParents, Bound<float>* aabbs, uint32_t* objIDs, int* flags, int numObjs) {

			const int tid = threadIdx.x + blockIdx.x * blockDim.x;
			if (tid >= numObjs) return;

			// Keep track of the maximum stack size required for DFS
			// Assuming full exploration and always pushing the left child first.
			int depth = 1;

			Bound<float> last = aabbs[objIDs[tid]];
			int parent = leafParents[tid];

			while (true) {
				int isRight = (parent & 0x80000000) != 0;
				parent = parent & 0x7FFFFFFF;
				nodes[parent].bounds[isRight] = last;

				// Exit if we are the first thread here
				int otherDepth = atomicOr(flags + parent, depth);
				if (!otherDepth) return;

				if (isRight)
					depth = std::max(depth + 1, otherDepth);
				else
					depth = std::max(depth, otherDepth + 1);

				// Ensure memory coherency before we read.
				__threadfence();

				if (!parent) {			// We've reached the root.
					flags[0] = depth;	// Only the one lucky thread gets to finish up.
					return;
				}
				last.absorb(nodes[parent].bounds[1 - isRight]);
				parent = nodes[parent].parentIdx;
			}
		}
		
		// Calculates subtree sizes bottom-up
		__global__ void calcSubtreeSizesKernel(LBVHNode<float>* nodes,
			uint32_t* leafParents, int* flags, int* subtreeSizes, int numObjs) {

			const int tid = threadIdx.x + blockIdx.x * blockDim.x;
			if (tid >= numObjs) return;

			int parent = leafParents[tid];
			int size = 1; // Leaf size is 1

			while (true) {
				// parent includes MSB for left/right, strip it
				int isRight = (parent & 0x80000000) != 0;
				parent = parent & 0x7FFFFFFF;
				
				// We can reuse the 'flags' array to store the size of the first child to arrive
				
				int oldVal = atomicExch(flags + parent, size);
				
				if (oldVal == 0) {
					// First child arrived, wait for second
					return;
				}
				
				// Second child arrived. oldVal is the size of the sibling.
				// My size (parent size) = my child size + sibling size + 1 (for parent node itself)
				size = size + oldVal + 1;
				subtreeSizes[parent] = size;
				
				__threadfence();
				
				if (!parent) return; // Reached root
				
				parent = nodes[parent].parentIdx;
			}
		}

		
		// Compute pre-order indices and escape pointers directly, then write in parallel. Each thread computes its node's position by walking to root - O(depth) per thread
		
		__device__ inline int getSubtreeSize(int nodeIdx, int* subtreeSizes) {
			if (nodeIdx & 0x80000000) return 1;  // Leaf has size 1
			return subtreeSizes[nodeIdx];
		}
		
		// Kernel to compute pre-order index and escape for internal nodes
		__global__ void computeInternalNodeLayout(
			LBVHNode<float>* nodes,
			int* subtreeSizes,
			int* preorderIndices,   // Output: preorder index for each internal node
			int* escapeIndices,     // Output: escape index for each internal node
			int numInternalNodes) {
			
			const int tid = threadIdx.x + blockIdx.x * blockDim.x;
			if (tid >= numInternalNodes) return;
			
			int preorder = 0;
			int escapePreorder = -1;
			int nodeIdx = tid;
			
			//root nodeIdx = 0
			while (nodeIdx != 0) {
				uint32_t parentInfo = nodes[nodeIdx].parentIdx;
				bool isRight = (parentInfo & 0x80000000) != 0;
				int parentIdx = parentInfo & 0x7FFFFFFF;
				
				if (isRight) {
					// Right child: add 1 (for parent) + left sibling's subtree size
					int leftIdx = nodes[parentIdx].leftIdx;
					preorder += 1 + getSubtreeSize(leftIdx, subtreeSizes);
				} else {
					// Left child: just add 1 for parent
					preorder += 1;
					// First left-child ancestor determines escape (its right sibling)
					if (escapePreorder == -1) {
						int rightIdx = nodes[parentIdx].rightIdx;
						// We'll compute right sibling's preorder later, store parent for now
						escapePreorder = parentIdx; 
					}
				}
				nodeIdx = parentIdx;
			}
			
			preorderIndices[tid] = preorder;
			escapeIndices[tid] = escapePreorder;  // Temporarily stores info for escape resolution
		}
		
		// Kernel to compute pre-order index and escape for leaf nodes
		__global__ void computeLeafNodeLayout(
			LBVHNode<float>* nodes,
			uint32_t* leafParents,
			int* subtreeSizes,
			int* leafPreorderIndices,  // Output
			int* leafEscapeIndices,    // Output
			int numObjs) {
			
			const int tid = threadIdx.x + blockIdx.x * blockDim.x;
			if (tid >= numObjs) return;
			
			int preorder = 0;
			int escapePreorder = -1;
			
			// Start from leaf's parent
			uint32_t parentInfo = leafParents[tid];
			bool isRight = (parentInfo & 0x80000000) != 0;
			int parentIdx = parentInfo & 0x7FFFFFFF;
			
			if (isRight) {
				int leftIdx = nodes[parentIdx].leftIdx;
				preorder += 1 + getSubtreeSize(leftIdx, subtreeSizes);
			} else {
				preorder += 1;
				escapePreorder = parentIdx;  // First left ancestor
			}
			
			// Walk up to root
			while (parentIdx != 0) {
				uint32_t gpInfo = nodes[parentIdx].parentIdx;
				bool gpIsRight = (gpInfo & 0x80000000) != 0;
				int gpIdx = gpInfo & 0x7FFFFFFF;
				
				if (gpIsRight) {
					int leftIdx = nodes[gpIdx].leftIdx;
					preorder += 1 + getSubtreeSize(leftIdx, subtreeSizes);
				} else {
					preorder += 1;
					if (escapePreorder == -1) {
						escapePreorder = gpIdx;
					}
				}
				parentIdx = gpIdx;
			}
			
			leafPreorderIndices[tid] = preorder;
			leafEscapeIndices[tid] = escapePreorder;
		}
		
		// Kernel to resolve escape pointers and write internal nodes
		__global__ void writeInternalStacklessNodes(
			LBVHNode<float>* nodes,
			StacklessNode* stacklessNodes,
			int* subtreeSizes,
			int* preorderIndices,
			int* escapeIndices,
			int* leafPreorderIndices,
			int numInternalNodes) {
			
			const int tid = threadIdx.x + blockIdx.x * blockDim.x;
			if (tid >= numInternalNodes) return;
			
			int preorder = preorderIndices[tid];
			int escapeParent = escapeIndices[tid];
			
			// Resolve escape: find right sibling's preorder index
			int escape = -1;
			if (escapeParent >= 0) {
				int rightIdx = nodes[escapeParent].rightIdx;
				if (rightIdx & 0x80000000) {
					// Right sibling is a leaf
					escape = leafPreorderIndices[rightIdx & 0x7FFFFFFF];
				} else {
					// Right sibling is internal
					escape = preorderIndices[rightIdx];
				}
			}
			
			// Build the stackless node
			LBVHNode<float> node = nodes[tid];
			StacklessNode snode;
			snode.bound = node.bounds[0];
			snode.bound.absorb(node.bounds[1]);
			snode.data = -1;  // mark internal node by setting data = -1
			snode.escape = escape;
			
			stacklessNodes[preorder] = snode;
		}
		
		__global__ void writeLeafStacklessNodes(
			LBVHNode<float>* nodes,
			StacklessNode* stacklessNodes,
			Bound<float>* objAabbs,
			uint32_t* objIDs,
			int* preorderIndices,
			int* leafPreorderIndices,
			int* leafEscapeIndices,
			int numObjs) {
			
			const int tid = threadIdx.x + blockIdx.x * blockDim.x;
			if (tid >= numObjs) return;
			
			int preorder = leafPreorderIndices[tid];
			int escapeParent = leafEscapeIndices[tid];
			
			// Resolve escape
			int escape = -1;
			if (escapeParent >= 0) {
				int rightIdx = nodes[escapeParent].rightIdx;
				if (rightIdx & 0x80000000) {
					escape = leafPreorderIndices[rightIdx & 0x7FFFFFFF];
				} else {
					escape = preorderIndices[rightIdx];
				}
			}
			
			// Build the stackless node
			int objId = objIDs[tid];
			StacklessNode snode;
			snode.bound = objAabbs[objId];
			snode.data = objId;  // Leaf node's data stores primitive index
			snode.escape = escape;
			
			stacklessNodes[preorder] = snode;
		}

		// Query the LBVH using Stackless Traversal
		template<bool IGNORE_SELF>
		__global__ void lbvhStacklessQueryKernel(int2* res, int* resCounter, int maxRes,
			const StacklessNode* nodes, const int* subtreeSizes,
			const uint32_t* queryIDs, const Bound<float>* queryAABBs, const int numQueries) {

			const int tid = threadIdx.x + blockIdx.x * blockDim.x;
			bool active = tid < numQueries;

			Bound<float> queryAABB;
			int objIdx;
			if (active) {
				objIdx = queryIDs[tid];
				queryAABB = queryAABBs[objIdx];
			}

		__shared__ int2 sharedRes[MAX_RES_PER_BLOCK];
		__shared__ int sharedCounter;		
		__shared__ int sharedGlobalIdx;
		__shared__ int sharedActiveCount;  // Track how many threads are still active
		
		if (threadIdx.x == 0) {
			sharedCounter = 0;
			sharedActiveCount = 0;
		}
		__syncthreads();
		
		if (active) atomicAdd(&sharedActiveCount, 1);
		__syncthreads();

		int nodeIdx = 0; 
		bool threadDone = !active; // inactive threads are already done
		
		while (true) {
			__syncthreads();

			if (!threadDone) {
				// Loop until we find a collision or finish tree (nodeIdx == -1)
				
				int loopCount = 0;
				while (nodeIdx != -1 && loopCount < 100) { // Limit inner loop to allow flushing
					loopCount++;
					
					StacklessNode node = nodes[nodeIdx];
					
					bool hit = node.bound.intersects(queryAABB);
					
					// Stackless tranversal logic
					if (hit) {
						//node date is -1 for internal nodes, primitive index for leaf nodes
						if (node.data != -1) {
							// LEAF NODE
							
							// collision logic
							bool report = true;
							if (IGNORE_SELF && node.data == objIdx) report = false;
							
							if (report) {
								int sIdx = atomicAdd(&sharedCounter, 1);
								if (sIdx < MAX_RES_PER_BLOCK) {
									sharedRes[sIdx] = make_int2(node.data, objIdx);
								} else {
									atomicSub(&sharedCounter, 1); // Revert count
									break; // Break inner loop, nodeIdx is NOT updated
								}
							}
							
							// Finished leaf, go to escape
							nodeIdx = node.escape;
						} else {
							// INTERNAL NODE
							// Go to left child (which is +1 in preorder)
							nodeIdx = nodeIdx + 1;
						}
					} else {
						// MISS: no intersection
						// Skip subtree-bv safely
						nodeIdx = node.escape;
					}
				}
				
				// Check if this thread finished traversal
				if (nodeIdx == -1) {
					threadDone = true;
					atomicSub(&sharedActiveCount, 1);
				}
			}
			
			// flush whatever we have
			__syncthreads();
			int totalRes = min(sharedCounter, MAX_RES_PER_BLOCK);
			int activeCount = sharedActiveCount;

			if (threadIdx.x == 0)
				sharedGlobalIdx = atomicAdd(resCounter, totalRes);

			__syncthreads();

			const int globalIdx = sharedGlobalIdx;
			
			if (totalRes > 0) {
				if (threadIdx.x == 0) sharedCounter = 0;
				
				// Copy to global
				int writeCount = totalRes;
				if (writeCount > maxRes - globalIdx) {
					writeCount = maxRes - globalIdx;
				}

				int fullBlocks = (writeCount - 1) / (int)blockDim.x;
				for (int i = 0; i < fullBlocks; i++) {
					int idx = i * blockDim.x + threadIdx.x;
					res[globalIdx + idx] = sharedRes[idx];
				}
				int idx = fullBlocks * blockDim.x + threadIdx.x;
				if (idx < writeCount) res[globalIdx + idx] = sharedRes[idx];
			}
			
			__syncthreads();
			
			// All threads exit together when no one is active
			if (activeCount == 0 || globalIdx >= maxRes) {
				return;
			}
		}
	}

}//namespace culbvh

#pragma endregion
#pragma region LBVHStacklessLite
	LBVHStacklessLite::LBVHStacklessLite() : impl(std::make_unique<thrustImpl>()) {}
	culbvh::LBVHStacklessLite::~LBVHStacklessLite() = default;

	LBVHStacklessLite::aabb LBVHStacklessLite::bounds() const {
		return rootBounds;
	}

	const thrust::device_vector<LBVHNode<float>>& LBVHStacklessLite::internal_nodes() const {
		return impl->d_nodes;
	}

	const thrust::device_ptr<LBVHStacklessLite::aabb>& LBVHStacklessLite::object_aabbs() const {
		return impl->d_objs;
	}

	void LBVHStacklessLite::refit() {
		cudaMemset(thrust::raw_pointer_cast(impl->d_flags.data()), 0, sizeof(uint32_t) * (numObjs - 1));

		// Go through and merge all the aabbs up from the leaf nodes
		LBVHStacklessLiteKernels::mergeUpKernel << <(numObjs + 255) / 256, 256 >> > (
			thrust::raw_pointer_cast(impl->d_nodes.data()),
			thrust::raw_pointer_cast(impl->d_leafParents.data()),
			thrust::raw_pointer_cast(impl->d_objs),
			thrust::raw_pointer_cast(impl->d_objIDs.data()),
			thrust::raw_pointer_cast(impl->d_flags.data()), numObjs);
			
		const unsigned int numInternalNodes = numObjs - 1;
		
		// Preorder indices and escape don't change during refit, only bounds. So we just need to rewrite the nodes with updated bounds.
		LBVHStacklessLiteKernels::writeInternalStacklessNodes<<<(numInternalNodes + 255) / 256, 256>>>(
			thrust::raw_pointer_cast(impl->d_nodes.data()),
			thrust::raw_pointer_cast(impl->d_stackless_nodes.data()),
			thrust::raw_pointer_cast(impl->d_subtree_sizes.data()),
			thrust::raw_pointer_cast(impl->d_internalPreorder.data()),
			thrust::raw_pointer_cast(impl->d_internalEscape.data()),
			thrust::raw_pointer_cast(impl->d_leafPreorder.data()),
			numInternalNodes
		);
		
		LBVHStacklessLiteKernels::writeLeafStacklessNodes<<<(numObjs + 255) / 256, 256>>>(
			thrust::raw_pointer_cast(impl->d_nodes.data()),
			thrust::raw_pointer_cast(impl->d_stackless_nodes.data()),
			thrust::raw_pointer_cast(impl->d_objs),
			thrust::raw_pointer_cast(impl->d_objIDs.data()),
			thrust::raw_pointer_cast(impl->d_internalPreorder.data()),
			thrust::raw_pointer_cast(impl->d_leafPreorder.data()),
			thrust::raw_pointer_cast(impl->d_leafEscape.data()),
			numObjs
		);

		checkCudaErrors(cudaGetLastError());
	}

	void LBVHStacklessLite::compute(aabb* devicePtr, size_t size) {
		impl->d_objs = thrust::device_ptr<aabb>(devicePtr);
		numObjs = size;

		const unsigned int numInternalNodes = numObjs - 1;	// Total number of internal nodes
		const unsigned int numNodes = numObjs * 2 - 1;		// Total number of nodes

		impl->d_morton.resize(numObjs);
		impl->d_objIDs.resize(numObjs);
		impl->d_leafParents.resize(numObjs);
		impl->d_nodes.resize(numInternalNodes);
		impl->d_flags.resize(numInternalNodes);
	
		impl->d_stackless_nodes.resize(numNodes);
		impl->d_subtree_sizes.resize(numNodes);
		
		// Preorder layout indices
		impl->d_internalPreorder.resize(numInternalNodes);
		impl->d_internalEscape.resize(numInternalNodes);
		impl->d_leafPreorder.resize(numObjs);
		impl->d_leafEscape.resize(numObjs);

		thrust::fill(impl->d_flags.begin(), impl->d_flags.end(), 0);

		rootBounds = aabb();
		rootBounds = thrust::reduce(
			impl->d_objs, impl->d_objs + numObjs, rootBounds,
			[] __host__ __device__(const aabb & lhs, const aabb & rhs) {
			auto b = lhs;
			b.absorb(rhs);
			return b;
		});

		LBVHStacklessLiteKernels::mortonKernel<float> << <(numObjs + 255) / 256, 256 >> > (
			devicePtr, thrust::raw_pointer_cast(impl->d_morton.data()),
			thrust::raw_pointer_cast(impl->d_objIDs.data()), rootBounds, numObjs);

		thrust::stable_sort_by_key(impl->d_morton.begin(), impl->d_morton.end(), impl->d_objIDs.begin());

		LBVHStacklessLiteKernels::lbvhBuildInternalKernel << <(numInternalNodes + 255) / 256, 256 >> > (
			thrust::raw_pointer_cast(impl->d_nodes.data()),
			thrust::raw_pointer_cast(impl->d_leafParents.data()),
			thrust::raw_pointer_cast(impl->d_morton.data()),
			thrust::raw_pointer_cast(impl->d_objIDs.data()), numObjs);
			
		LBVHStacklessLiteKernels::mergeUpKernel << <(numObjs + 255) / 256, 256 >> > (
			thrust::raw_pointer_cast(impl->d_nodes.data()),
			thrust::raw_pointer_cast(impl->d_leafParents.data()),
			thrust::raw_pointer_cast(impl->d_objs),
			thrust::raw_pointer_cast(impl->d_objIDs.data()),
			thrust::raw_pointer_cast(impl->d_flags.data()), numObjs);
			
		// Calculate Subtree Sizes
		thrust::fill(impl->d_flags.begin(), impl->d_flags.end(), 0);
		LBVHStacklessLiteKernels::calcSubtreeSizesKernel<<<(numObjs + 255)/256, 256>>>(
			thrust::raw_pointer_cast(impl->d_nodes.data()),
			thrust::raw_pointer_cast(impl->d_leafParents.data()),
			thrust::raw_pointer_cast(impl->d_flags.data()),
			thrust::raw_pointer_cast(impl->d_subtree_sizes.data()),
			numObjs
		);
		
		// Build Stackless Linear Layout
		// Step 1: Compute preorder indices and escape info for internal nodes
		LBVHStacklessLiteKernels::computeInternalNodeLayout<<<(numInternalNodes + 255) / 256, 256>>>(
			thrust::raw_pointer_cast(impl->d_nodes.data()),
			thrust::raw_pointer_cast(impl->d_subtree_sizes.data()),
			thrust::raw_pointer_cast(impl->d_internalPreorder.data()),
			thrust::raw_pointer_cast(impl->d_internalEscape.data()),
			numInternalNodes
		);
		
		// Step 2: Compute preorder indices and escape info for leaf nodes
		LBVHStacklessLiteKernels::computeLeafNodeLayout<<<(numObjs + 255) / 256, 256>>>(
			thrust::raw_pointer_cast(impl->d_nodes.data()),
			thrust::raw_pointer_cast(impl->d_leafParents.data()),
			thrust::raw_pointer_cast(impl->d_subtree_sizes.data()),
			thrust::raw_pointer_cast(impl->d_leafPreorder.data()),
			thrust::raw_pointer_cast(impl->d_leafEscape.data()),
			numObjs
		);
		
		// Step 3: Write internal stackless nodes in parallel
		LBVHStacklessLiteKernels::writeInternalStacklessNodes<<<(numInternalNodes + 255) / 256, 256>>>(
			thrust::raw_pointer_cast(impl->d_nodes.data()),
			thrust::raw_pointer_cast(impl->d_stackless_nodes.data()),
			thrust::raw_pointer_cast(impl->d_subtree_sizes.data()),
			thrust::raw_pointer_cast(impl->d_internalPreorder.data()),
			thrust::raw_pointer_cast(impl->d_internalEscape.data()),
			thrust::raw_pointer_cast(impl->d_leafPreorder.data()),
			numInternalNodes
		);
		
		// Step 4: Write leaf stackless nodes in parallel
		LBVHStacklessLiteKernels::writeLeafStacklessNodes<<<(numObjs + 255) / 256, 256>>>(
			thrust::raw_pointer_cast(impl->d_nodes.data()),
			thrust::raw_pointer_cast(impl->d_stackless_nodes.data()),
			thrust::raw_pointer_cast(impl->d_objs),
			thrust::raw_pointer_cast(impl->d_objIDs.data()),
			thrust::raw_pointer_cast(impl->d_internalPreorder.data()),
			thrust::raw_pointer_cast(impl->d_leafPreorder.data()),
			thrust::raw_pointer_cast(impl->d_leafEscape.data()),
			numObjs
		);
		
		checkCudaErrors(cudaGetLastError());
	}


	size_t LBVHStacklessLite::query(int2* d_res, size_t resSize, aabb* d_otherAABBs, size_t otherSize) const {
		thrust::device_ptr<aabb> d_otherAABBs_ptr(d_otherAABBs);
		thrust::device_vector<uint32_t> d_other_morton(otherSize);
		thrust::device_vector<uint32_t> d_other_objIDs(otherSize);
		aabb query_scene_box = aabb();
		// compute scene box
		query_scene_box = thrust::reduce(
			d_otherAABBs_ptr, d_otherAABBs_ptr + otherSize, query_scene_box,
			[] __host__ __device__(const aabb & lhs, const aabb & rhs) {
			auto b = lhs;
			b.absorb(rhs);
			return b;
		});

		// we query against morton order to allow maximized potential memory coalescing
		LBVHStacklessLiteKernels::mortonKernel<float> << <(otherSize + 255) / 256, 256 >> > (
			d_otherAABBs, thrust::raw_pointer_cast(d_other_morton.data()),
			thrust::raw_pointer_cast(d_other_objIDs.data()), query_scene_box, otherSize);
		thrust::sort_by_key(d_other_morton.begin(), d_other_morton.end(), d_other_objIDs.begin());


		int* d_counter = (int*)thrust::raw_pointer_cast(impl->d_flags.data());
		cudaMemset(d_counter, 0, sizeof(int));

		const int numQuery = numObjs;
		const int numOther = otherSize;
		
		printf("Stackless Query info: numQuery=%d, numOther=%d, resSize=%zu\n", numQuery, numOther, resSize);
		
		// the other objIDs is now in morton order
		LBVHStacklessLiteKernels::lbvhStacklessQueryKernel<false> << <(numOther + 127) / 128, 128 >> > (
			d_res, d_counter, resSize,
			thrust::raw_pointer_cast(impl->d_stackless_nodes.data()),
			thrust::raw_pointer_cast(impl->d_subtree_sizes.data()),
			thrust::raw_pointer_cast(d_other_objIDs.data()),
			d_otherAABBs,  // Already a raw pointer, no need for raw_pointer_cast
			numOther
		);

		checkCudaErrors(cudaGetLastError());
		
		int h_counter;
		cudaMemcpy(&h_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
		checkCudaErrors(cudaGetLastError());
		
		if (h_counter > resSize) {
			printf("Warning: Buffer overflow detected. Found %d collisions but buffer size is %zu\n", 
				h_counter, resSize);
		}
		
		return std::min((size_t)h_counter, resSize);
	}

	size_t LBVHStacklessLite::query(int2* d_res, size_t resSize) const {
		int* d_counter = (int*)thrust::raw_pointer_cast(impl->d_flags.data());
		cudaMemset(d_counter, 0, sizeof(int));

		const int numQuery = numObjs;
		
		printf("Stackless Self-query info: numQuery=%d, resSize=%zu\n", numQuery, resSize);
		
		LBVHStacklessLiteKernels::lbvhStacklessQueryKernel<true> << <(numQuery + 127) / 128, 128 >> > (
			d_res, d_counter, resSize,
			thrust::raw_pointer_cast(impl->d_stackless_nodes.data()),
			thrust::raw_pointer_cast(impl->d_subtree_sizes.data()),
			thrust::raw_pointer_cast(impl->d_objIDs.data()),
			thrust::raw_pointer_cast(impl->d_objs),
			numQuery
		);

		checkCudaErrors(cudaGetLastError());
		
		int h_counter;
		cudaMemcpy(&h_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
		checkCudaErrors(cudaGetLastError());
		
		if (h_counter > resSize) {
			printf("Warning: Buffer overflow detected. Found %d collisions but buffer size is %zu\n", 
				h_counter, resSize);
		}
		
		return std::min((size_t)h_counter, resSize);
	}

	bool LBVHStacklessLite::query_compare_ground_truth(int2* d_res, size_t resSize) const {
		thrust::device_ptr<int2> d_results_ptr(d_res);
		thrust::host_vector<int2> h_res(d_results_ptr, d_results_ptr + resSize);
		
		tbb::concurrent_unordered_set<int2, Int2Hash, Int2Equal> gpu_results;
		for (const auto& pair : h_res) {
			int2 ordered_pair = pair;
			if (ordered_pair.x > ordered_pair.y) {
				std::swap(ordered_pair.x, ordered_pair.y);
			}
			gpu_results.insert(ordered_pair);
		}

		thrust::host_vector<aabb> h_aabbs(impl->d_objs, impl->d_objs + numObjs);
		tbb::concurrent_unordered_set<int2, Int2Hash, Int2Equal> cpu_results;
		
		tbb::parallel_for(tbb::blocked_range<size_t>(0, numObjs),
			[&](const tbb::blocked_range<size_t>& range) {
				for (size_t i = range.begin(); i < range.end(); i++) {
					for (size_t j = i + 1; j < numObjs; j++) {
						if (h_aabbs[i].intersects(h_aabbs[j])) {
							cpu_results.insert(make_int2(i, j));
						}
					}
				}
			}
		);

		if (cpu_results.size() != gpu_results.size()) {
			printf("Error: Number of collision pairs mismatch. CPU: %zu, GPU: %zu\n", 
				   cpu_results.size(), gpu_results.size());
			return false;
		}

		for (const auto& pair : cpu_results) {
			if (gpu_results.find(pair) == gpu_results.end()) {
				printf("Error: CPU result (%d, %d) not found in GPU results\n", pair.x, pair.y);
				return false;
			}
		}
		for (const auto& pair : gpu_results) {
			if (cpu_results.find(pair) == cpu_results.end()) {
				printf("Error: GPU result (%d, %d) not found in CPU results\n", pair.x, pair.y);
				return false;
			}
		}

		return true;
	}

	bool LBVHStacklessLite::query_compare_ground_truth(int2* d_res, size_t resSize, aabb* d_otherAABBs, size_t otherSize) const {
		thrust::device_ptr<int2> d_results_ptr(d_res);
		thrust::host_vector<int2> h_res(d_results_ptr, d_results_ptr + resSize);
		thrust::device_ptr<aabb> d_otherAABBs_ptr(d_otherAABBs);
		thrust::host_vector<aabb> h_otherAABBs(d_otherAABBs_ptr, d_otherAABBs_ptr + otherSize);
		tbb::concurrent_unordered_set<int2, Int2Hash, Int2Equal> gpu_results;
		for (const auto& pair : h_res) {
			gpu_results.insert(pair);
		}
		thrust::host_vector<aabb> h_aabbs1(impl->d_objs, impl->d_objs + numObjs);
		
		tbb::concurrent_unordered_set<int2, Int2Hash, Int2Equal> cpu_results;

		tbb::parallel_for(tbb::blocked_range<size_t>(0, numObjs),
			[&](const tbb::blocked_range<size_t>& range) {
				for (size_t i = range.begin(); i < range.end(); i++) {
					for (size_t j = 0; j < otherSize; j++) {
						if (h_aabbs1[i].intersects(h_otherAABBs[j])) {
							cpu_results.insert(make_int2(i, j));
						}
					}
				}
			}
		);
		if (cpu_results.size() != resSize) {
			printf("Error: Number of collision pairs mismatch. CPU: %zu, GPU: %zu\n", 
				   cpu_results.size(), resSize);
			return false;
		}

		for (const auto& pair : cpu_results) {
			if (gpu_results.find(pair) == gpu_results.end()) {
				printf("Error: CPU result (%d, %d) not found in GPU results\n", pair.x, pair.y);
				return false;
			}
		}

		for (const auto& pair : gpu_results) {
			if (cpu_results.find(pair) == cpu_results.end()) {
				printf("Error: GPU result (%d, %d) not found in CPU results\n", pair.x, pair.y);
				return false;
			}
		}

		return true;
	}

#pragma endregion
#pragma region testing
namespace LBVHStacklessLiteTesting {

	void testAABBMatch(culbvh::Bound<float> a, culbvh::Bound<float> b, int idx) {
		if (squaredLength(a.min - b.min) > 1e-7f || squaredLength(a.max - b.max) > 1e-7f) {
			printf("Error: AABB mismatch node %d\n", idx);
			printf("Expected:\n");
			culbvh::print(a.min);
			culbvh::print(a.max);
			printf("Found:\n");
			culbvh::print(b.min);
			culbvh::print(b.max);
		}
	}

	culbvh::Bound<float> lbvhCheckAABBMerge(
		thrust::host_vector<LBVHNode<float>>& nodes,
		uint32_t idx) {
		auto node = nodes[idx];
		if (!(node.leftIdx >> 31)) {
			auto o = lbvhCheckAABBMerge(nodes, node.leftIdx);
			testAABBMatch(o, node.bounds[0], node.leftIdx & 0x7FFFFFFF);
		}
		if (!(node.rightIdx >> 31)) {
			auto o = lbvhCheckAABBMerge(nodes, node.rightIdx);
			testAABBMatch(o, node.bounds[1], node.rightIdx & 0x7FFFFFFF);
		}

		node.bounds[0].absorb(node.bounds[1]);
		return node.bounds[0];
	}

	int2 lbvhCheckIndexMerge(
		thrust::host_vector<LBVHNode<float>>& nodes,
		uint32_t idx, int numObjs) {
		bool isLeaf = idx >> 31;
		idx &= 0x7FFFFFFF;

		if (isLeaf) return make_int2(idx, idx);
		auto node = nodes[idx];
		int2 range = make_int2(idx, node.fence);
		if (range.y < range.x) std::swap(range.x, range.y);

		auto left = lbvhCheckIndexMerge(nodes, node.leftIdx, numObjs);
		auto right = lbvhCheckIndexMerge(nodes, node.rightIdx, numObjs);
		left.x = std::min(left.x, right.x);
		left.y = std::max(left.y, right.y);

		if (left != range)
			printf("Error: Index range mismatch\n");

		return left;
	}

}
	void LBVHStacklessLite::bvhSelfCheck() const {
		printf("\nLBVHStacklessLite self check...\n");

		thrust::host_vector<LBVHNode<float>> nodes(impl->d_nodes.begin(), impl->d_nodes.end());
		thrust::host_vector<uint32_t> morton(impl->d_morton.begin(), impl->d_morton.end());
		thrust::host_vector<uint32_t> leafParent(impl->d_leafParents.begin(), impl->d_leafParents.end());

		if (nodes.size() != numObjs - 1 || morton.size() != numObjs)
			printf("Error: Incorrect memory sizes\n");

		for (size_t i = 1; i < numObjs; i++)
			if (morton[i - 1] > morton[i])
				printf("Bad morton code ordering\n");

		for (size_t i = 1; i < numObjs - 1; i++) {
			auto node = nodes[i];
			uint32_t isRight = node.parentIdx >> 31;
			uint32_t parentIdx = node.parentIdx & 0x7FFFFFFF;
			auto parent = nodes[parentIdx];

			if ((isRight ? parent.rightIdx : parent.leftIdx) != i)
				printf("Error: Child node has incorrect parent\n");
		}

		for (size_t i = 0; i < numObjs; i++) {
			uint32_t parentIdx = leafParent[i];
			uint32_t isRight = parentIdx >> 31;
			parentIdx &= 0x7FFFFFFF;
			auto parent = nodes[parentIdx];

			if ((isRight ? parent.rightIdx : parent.leftIdx) != (i | 0x80000000))
				printf("Error: Leaf node has incorrect parent\n");
		}

		LBVHStacklessLiteTesting::lbvhCheckIndexMerge(nodes, 0, numObjs);
		LBVHStacklessLiteTesting::testAABBMatch(LBVHStacklessLiteTesting::lbvhCheckAABBMerge(nodes, 0u), rootBounds, 0);

		printf("LBVHStacklessLite self check complete.\n\n");
	}

	void testLBVHStacklessLite() {
		const int N = 200000;
		const float R = 0.001f;

		printf("Generating Data...\n");
		vector<culbvh::Bound<float>> points(N);

		srand(1);
		for (size_t i = 0; i < N; i++) {
			culbvh::Bound<float> b(make_float3(rand() / (float)RAND_MAX, rand() / (float)RAND_MAX, rand() / (float)RAND_MAX));
			b.pad(R);
			points[i] = b;
		}

		thrust::device_vector<culbvh::Bound<float>> d_points(points.begin(), points.end());
		thrust::device_vector<int2> d_res(100 * N);
		cudaDeviceSynchronize();

		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		float milliseconds = 0;

		printf("Building LBVH...\n");
		cudaEventRecord(start);
		LBVHStacklessLite bvh;
		bvh.compute(thrust::raw_pointer_cast(d_points.data()), N);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("LBVH build time: %.3f ms\n", milliseconds);

		printf("Querying LBVH...\n");
		cudaEventRecord(start);
		int numCols = bvh.query(thrust::raw_pointer_cast(d_res.data()), d_res.size());
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("LBVH query time: %.3f ms\n", milliseconds);

		printf("Getting results...\n");
		thrust::host_vector<int2> res(d_res.begin(), d_res.begin() + numCols);

		printf("%d collision pairs found on GPU.\n", res.size());

		tbb::concurrent_unordered_set<int2, Int2Hash, Int2Equal> resSet;
		bool good = true;

		for (size_t i = 0; i < res.size(); i++) {
			int2 a = res[i];
			if (a.x > a.y) std::swap(a.x, a.y);
			if (!resSet.insert(a).second) {
				printf("Error: Duplicate result\n");
				good = false;
			}
		}

		std::atomic<int> numCPUFound{0};

		printf("\nRunning brute force CPU collision detection with TBB...\n");
		
		auto cpu_start = std::chrono::high_resolution_clock::now();
		
		tbb::parallel_for(tbb::blocked_range<int>(0, N),
			[&](const tbb::blocked_range<int>& range) {
				for (int i = range.begin(); i < range.end(); i++) {
					for (int j = i + 1; j < N; j++) {
						if (points[i].intersects(points[j])) {
							numCPUFound++;
							if (resSet.find(make_int2(i, j)) == resSet.end()) {
								printf("Error: CPU result %d %d not found in GPU result.\n", i, j);
								good = false;
							}
						}
					}
				}
			}
		);

		auto cpu_end = std::chrono::high_resolution_clock::now();
		auto cpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start);

		if (numCPUFound != res.size()) {
			printf("Error: CPU and GPU number of results do not match\n");
			good = false;
		}

		printf("%d collision pairs found on CPU.\n", numCPUFound.load());
		printf("CPU computation time: %lld ms\n", cpu_duration.count());
		printf(good ? "CPU and GPU results match.\n" : "CPU and GPU results MISMATCH!\n");

		bvh.bvhSelfCheck();

		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

#pragma endregion

}