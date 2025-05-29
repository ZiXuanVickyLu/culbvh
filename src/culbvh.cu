#include "culbvh.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

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

#include <unordered_set>
#include <tbb/parallel_for.h>
#include <atomic>
#include <tbb/concurrent_unordered_set.h>

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

	struct LBVH::thrustImpl {
		thrust::device_ptr<Bound<float>> d_objs = nullptr;
		thrust::device_vector<int> d_flags;				// Flags used for updating the tree

		thrust::device_vector<uint32_t> d_morton;		// Morton codes for each object
		thrust::device_vector<uint32_t> d_objIDs;		// Object ID for each leaf
		thrust::device_vector<uint32_t> d_leafParents;	// Parent ID for each leaf. MSB is used to indicate whether this is a left or right child of said parent.
		thrust::device_vector<LBVHNode<float>> d_nodes;			// The internal tree nodes
	};

#pragma region LBVHDevice

	namespace LBVHKernels {
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

		// Uses CUDA intrinsics for counting leading zeros
		__device__ inline int commonUpperBits(const uint64_t lhs, const uint64_t rhs) {
			return ::__clzll(lhs ^ rhs);
		}

		// Merges morton code with its index to output a sorted unique 64-bit key.
		__device__ inline uint64_t mergeIdx(const uint32_t code, const int idx) {
			return ((uint64_t)code << 32ul) | (uint64_t)idx;
		}

		__device__ inline int2 determineRange(uint32_t const* mortonCodes,
			const uint32_t numObjs, uint32_t idx) {

			// This is the root node
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

			// Left and right children are neighbors to the split point
			// Check if there are leaf nodes, which are indexed behind the (numObj - 1) internal nodes
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

		// Query the LBVH for overlapping objects
		// Overcomplicated because of shared memory buffering
		template<bool IGNORE_SELF, int STACK_SIZE>
		__global__ void lbvhQueryKernel(int2* res, int* resCounter, int maxRes,
			const LBVHNode<float>* nodes, const uint32_t* objIDs,
			const uint32_t* queryIDs, const Bound<float>* queryAABBs, const int numQueries) {

			const int tid = threadIdx.x + blockIdx.x * blockDim.x;
			bool active = tid < numQueries;

			Bound<float> queryAABB;
			int objIdx;
			if (active) {
				objIdx = queryIDs[tid];
				queryAABB = queryAABBs[objIdx];
				// if(!culbvh_aabb_valid_predicate<float>()(queryAABB)){
				// 	printf("Invalid queryAABB found at%d: %f,%f,%f,%f,%f,%f\n", objIdx, queryAABB.min.x, queryAABB.min.y, queryAABB.min.z, queryAABB.max.x, queryAABB.max.y, queryAABB.max.z);
				// }
			}

			__shared__ int2 sharedRes[MAX_RES_PER_BLOCK];
			__shared__ int sharedCounter;		// How many results are cached in shared memory
			__shared__ int sharedGlobalIdx;		// Where to write in global memory
			if (threadIdx.x == 0)
				sharedCounter = 0;

			uint32_t stack[STACK_SIZE];			// This is dynamically sized through templating
			uint32_t* stackPtr = stack;
			*(stackPtr++) = 0;					// Push

			while (true) {
				__syncthreads();

				if (active)
					while (stackPtr != stack) {
						uint32_t nodeIdx = *(--stackPtr);	// Pop
						bool isLeaf = nodeIdx & 0x80000000;
						nodeIdx = nodeIdx & 0x7FFFFFFF;

						if (isLeaf) {
							if (IGNORE_SELF)
								if (nodeIdx <= tid) continue;

							// Add to shared memory
							int sIdx = atomicAdd(&sharedCounter, 1);
							if (sIdx >= MAX_RES_PER_BLOCK) {
								// We cannot sync here so we push the node back on the stack and wait
								*(stackPtr++) = nodeIdx | 0x80000000;
								break;
							}
							sharedRes[sIdx] = make_int2(objIDs[nodeIdx], objIdx);
						}
						else {
							auto node = nodes[nodeIdx];

							// Ignore duplicate and self intersections
							if (IGNORE_SELF)
								if (std::max(nodeIdx, node.fence) <= tid) continue;

							// Internal node
							if (node.bounds[0].intersects(queryAABB))
								*(stackPtr++) = node.leftIdx;	// Push

							if (node.bounds[1].intersects(queryAABB))
								*(stackPtr++) = node.rightIdx;	// Push
						}
					}

				// Flush whatever we have
				__syncthreads();
				int totalRes = std::min(sharedCounter, MAX_RES_PER_BLOCK);

				if (threadIdx.x == 0)
					sharedGlobalIdx = atomicAdd(resCounter, totalRes);

				__syncthreads();

				// Make sure we dont write out of bounds
				const int globalIdx = sharedGlobalIdx;

				if (globalIdx >= maxRes || !totalRes) {
					//printf("Out of bounds\n");
					return;	// Out of memory for results.or there is no collision, safe exit
				}
				if (threadIdx.x == 0) sharedCounter = 0;

				// If we got here with a half full buffer, we are done.
				bool done = totalRes < MAX_RES_PER_BLOCK;
				// If we are about to run out of memory, we are done.
				if (totalRes > maxRes - globalIdx) {
					totalRes = maxRes - globalIdx;
					done = true;
				}

				// Copy full blocks
				int fullBlocks = (totalRes - 1) / (int)blockDim.x;
				for (int i = 0; i < fullBlocks; i++) {
					int idx = i * blockDim.x + threadIdx.x;
					res[globalIdx + idx] = sharedRes[idx];
				}

				// Copy the rest
				int idx = fullBlocks * blockDim.x + threadIdx.x;
				if (idx < totalRes) res[globalIdx + idx] = sharedRes[idx];

				// Break if every thread is done.
				if (done) break;
			}
		}

		// We are primarily limited by the number of registers, so we always call the kernel with just enough stack space.
		// This gives another ~15% performance boost for queries.
		template<bool IGNORE_SELF>
		void launchQueryKernel(int2* res, int* resCounter, int maxRes,
			const LBVHNode<float>* nodes, const uint32_t* objIDs,
			const uint32_t* queryIDs, const Bound<float>* queryAABBs, const int numQueries, int stackSize) {

			// This is a bit ugly but we want to compile the kernel for all stack sizes.
#define DISPATCH_QUERY(N) case N: lbvhQueryKernel<IGNORE_SELF, N> << <(numQueries + 127) / 128, 128 >> > (res, resCounter, maxRes, nodes, objIDs, queryIDs, queryAABBs, numQueries); break;
			switch (stackSize) {
			default:
				DISPATCH_QUERY(32); DISPATCH_QUERY(31); DISPATCH_QUERY(30); DISPATCH_QUERY(29); DISPATCH_QUERY(28); DISPATCH_QUERY(27); DISPATCH_QUERY(26); DISPATCH_QUERY(25);
				DISPATCH_QUERY(24); DISPATCH_QUERY(23); DISPATCH_QUERY(22); DISPATCH_QUERY(21); DISPATCH_QUERY(20); DISPATCH_QUERY(19); DISPATCH_QUERY(18); DISPATCH_QUERY(17);
				DISPATCH_QUERY(16); DISPATCH_QUERY(15); DISPATCH_QUERY(14); DISPATCH_QUERY(13); DISPATCH_QUERY(12); DISPATCH_QUERY(11); DISPATCH_QUERY(10); DISPATCH_QUERY(9);
				DISPATCH_QUERY(8); DISPATCH_QUERY(7); DISPATCH_QUERY(6); DISPATCH_QUERY(5); DISPATCH_QUERY(4); DISPATCH_QUERY(3); DISPATCH_QUERY(2); DISPATCH_QUERY(1);
			}
#undef DISPATCH_QUERY
		}
	}

#pragma endregion
#pragma region LBVH
	LBVH::LBVH() : impl(std::make_unique<thrustImpl>()) {}
	culbvh::LBVH::~LBVH() = default;

	LBVH::aabb LBVH::bounds() const {
		return rootBounds;
	}

	const thrust::device_vector<LBVHNode<float>>& LBVH::internal_nodes() const {
		return impl->d_nodes;
	}

	const thrust::device_ptr<LBVH::aabb>& LBVH::object_aabbs() const {
		return impl->d_objs;
	}

	void LBVH::refit() {
		cudaMemset(thrust::raw_pointer_cast(impl->d_flags.data()), 0, sizeof(uint32_t) * (numObjs - 1));

		// Go through and merge all the aabbs up from the leaf nodes
		LBVHKernels::mergeUpKernel << <(numObjs + 255) / 256, 256 >> > (
			thrust::raw_pointer_cast(impl->d_nodes.data()),
			thrust::raw_pointer_cast(impl->d_leafParents.data()),
			thrust::raw_pointer_cast(impl->d_objs),
			thrust::raw_pointer_cast(impl->d_objIDs.data()),
			thrust::raw_pointer_cast(impl->d_flags.data()), numObjs);

		checkCudaErrors(cudaGetLastError());
	}

	void LBVH::compute(aabb* devicePtr, size_t size) {
		impl->d_objs = thrust::device_ptr<aabb>(devicePtr);
		//this will make the BVH valid once compute is called
		numObjs = size;

		const unsigned int numInternalNodes = numObjs - 1;	// Total number of internal nodes
		const unsigned int numNodes = numObjs * 2 - 1;		// Total number of nodes

		impl->d_morton.resize(numObjs);
		impl->d_objIDs.resize(numObjs);
		impl->d_leafParents.resize(numObjs);
		impl->d_nodes.resize(numInternalNodes);
		impl->d_flags.resize(numInternalNodes);

		// Initialize flags to 0
		thrust::fill(impl->d_flags.begin(), impl->d_flags.end(), 0);

		// Compute the bounding box for the whole scene so we can assign morton codes
		rootBounds = aabb();
		rootBounds = thrust::reduce(
			impl->d_objs, impl->d_objs + numObjs, rootBounds,
			[] __host__ __device__(const aabb & lhs, const aabb & rhs) {
			auto b = lhs;
			b.absorb(rhs);
			return b;
		});

		// Compute morton codes. These don't have to be unique here.
		LBVHKernels::mortonKernel<float> << <(numObjs + 255) / 256, 256 >> > (
			devicePtr, thrust::raw_pointer_cast(impl->d_morton.data()),
			thrust::raw_pointer_cast(impl->d_objIDs.data()), rootBounds, numObjs);

		// Sort morton codes
		thrust::stable_sort_by_key(impl->d_morton.begin(), impl->d_morton.end(), impl->d_objIDs.begin());

		// Build out the internal nodes
		LBVHKernels::lbvhBuildInternalKernel << <(numInternalNodes + 255) / 256, 256 >> > (
			thrust::raw_pointer_cast(impl->d_nodes.data()),
			thrust::raw_pointer_cast(impl->d_leafParents.data()),
			thrust::raw_pointer_cast(impl->d_morton.data()),
			thrust::raw_pointer_cast(impl->d_objIDs.data()), numObjs);

		refit();
		
		// Ensure all CUDA operations are complete before reading the result
		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());
		
		// Copy max depth from device to host
		int h_max_depth;
		cudaMemcpy(&h_max_depth, thrust::raw_pointer_cast(impl->d_flags.data()), sizeof(int), cudaMemcpyDeviceToHost);
		checkCudaErrors(cudaGetLastError());
		maxStackSize = h_max_depth;		// Save max depth for query invocation
	}

	size_t LBVH::query(int2* d_res, size_t resSize, LBVH* other) const {
		// Borrow the flags array for the counter
		int* d_counter = (int*)thrust::raw_pointer_cast(impl->d_flags.data());
		cudaMemset(d_counter, 0, sizeof(int));

		// Query the LBVH
		const int numQuery = numObjs;
		const int numOther = other->numObjs;
		
		// Print diagnostic information
		printf("Query info: numQuery=%d, numOther=%d, resSize=%zu\n", numQuery, numOther, resSize);
		
		LBVHKernels::launchQueryKernel<false>(
			d_res, d_counter, resSize,
			thrust::raw_pointer_cast(impl->d_nodes.data()),
			thrust::raw_pointer_cast(impl->d_objIDs.data()),
			thrust::raw_pointer_cast(other->impl->d_objIDs.data()),
			thrust::raw_pointer_cast(other->impl->d_objs),
			numOther, maxStackSize
		);

		checkCudaErrors(cudaGetLastError());
		
		// Copy counter value from device to host
		int h_counter;
		cudaMemcpy(&h_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
		checkCudaErrors(cudaGetLastError());
		
		if (h_counter > resSize) {
			printf("Warning: Buffer overflow detected. Found %d collisions but buffer size is %zu\n", 
				h_counter, resSize);
		}
		
		return std::min((size_t)h_counter, resSize);
	}

	size_t LBVH::query(int2* d_res, size_t resSize) const {
		// Borrow the flags array for the counter
		int* d_counter = (int*)thrust::raw_pointer_cast(impl->d_flags.data());
		cudaMemset(d_counter, 0, sizeof(int));

		// Query the LBVH
		const int numQuery = numObjs;
		
		// Print diagnostic information
		printf("Self-query info: numQuery=%d, resSize=%zu\n", numQuery, resSize);
		
		LBVHKernels::launchQueryKernel<true>(
			d_res, d_counter, resSize,
			thrust::raw_pointer_cast(impl->d_nodes.data()),
			thrust::raw_pointer_cast(impl->d_objIDs.data()),
			thrust::raw_pointer_cast(impl->d_objIDs.data()),
			thrust::raw_pointer_cast(impl->d_objs),
			numQuery, maxStackSize
		);

		checkCudaErrors(cudaGetLastError());
		
		// Copy counter value from device to host
		int h_counter;
		cudaMemcpy(&h_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
		checkCudaErrors(cudaGetLastError());
		
		if (h_counter > resSize) {
			printf("Warning: Buffer overflow detected. Found %d collisions but buffer size is %zu\n", 
				h_counter, resSize);
		}
		
		return std::min((size_t)h_counter, resSize);
	}

	bool LBVH::query_compare_ground_truth(int2* d_res, size_t resSize) const {
		// Create device pointer from raw pointer
		thrust::device_ptr<int2> d_results_ptr(d_res);
		// Copy to host
		thrust::host_vector<int2> h_res(d_results_ptr, d_results_ptr + resSize);
		
		// Create a set to store unique collision pairs
		tbb::concurrent_unordered_set<int2, Int2Hash, Int2Equal> gpu_results;
		for (const auto& pair : h_res) {
			// Ensure pairs are ordered (i < j) to match GPU query behavior
			int2 ordered_pair = pair;
			if (ordered_pair.x > ordered_pair.y) {
				std::swap(ordered_pair.x, ordered_pair.y);
			}
			gpu_results.insert(ordered_pair);
		}

		thrust::host_vector<aabb> h_aabbs(impl->d_objs, impl->d_objs + numObjs);
		tbb::concurrent_unordered_set<int2, Int2Hash, Int2Equal> cpu_results;
		
		// Use TBB parallel_for for brute force check
		// Only check pairs where i < j to match GPU query behavior
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

	bool LBVH::query_compare_ground_truth(int2* d_res, size_t resSize, LBVH* other) const {
		// Create device pointer from raw pointer
		thrust::device_ptr<int2> d_results_ptr(d_res);
		// Copy to host
		thrust::host_vector<int2> h_res(d_results_ptr, d_results_ptr + resSize);
		
		tbb::concurrent_unordered_set<int2, Int2Hash, Int2Equal> gpu_results;
		for (const auto& pair : h_res) {
			gpu_results.insert(pair);  // No need to order pairs for cross-BVH comparison
		}
		thrust::host_vector<aabb> h_aabbs1(impl->d_objs, impl->d_objs + numObjs);
		thrust::host_vector<aabb> h_aabbs2(other->impl->d_objs, other->impl->d_objs + other->numObjs);
		
		tbb::concurrent_unordered_set<int2, Int2Hash, Int2Equal> cpu_results;

		tbb::parallel_for(tbb::blocked_range<size_t>(0, numObjs),
			[&](const tbb::blocked_range<size_t>& range) {
				for (size_t i = range.begin(); i < range.end(); i++) {
					for (size_t j = 0; j < other->numObjs; j++) {
						if (h_aabbs1[i].intersects(h_aabbs2[j])) {
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

	void testAABBMatch(culbvh::Bound<float> a, culbvh::Bound<float> b, int idx) {
		// Check if they match
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

		// Check if they match
		if (left != range)
			printf("Error: Index range mismatch\n");

		return left;
	}

	void LBVH::bvhSelfCheck() const {
		printf("\nLBVH self check...\n");

		// Get nodes
		thrust::host_vector<LBVHNode<float>> nodes(impl->d_nodes.begin(), impl->d_nodes.end());
		thrust::host_vector<uint32_t> morton(impl->d_morton.begin(), impl->d_morton.end());
		thrust::host_vector<uint32_t> leafParent(impl->d_leafParents.begin(), impl->d_leafParents.end());

		// Check sizes
		if (nodes.size() != numObjs - 1 || morton.size() != numObjs)
			printf("Error: Incorrect memory sizes\n");

		// Check morton codes
		for (size_t i = 1; i < numObjs; i++)
			if (morton[i - 1] > morton[i])
				printf("Bad morton code ordering\n");

		// Check that all children have the correct parent
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

		// Check that all nodes are accessible from the root
		std::vector<uint32_t> stack;
		int numVisited = 0;
		int maxSize = 0;
		stack.push_back(0);
		while (stack.size()) {
			auto idx = stack.back();
			stack.pop_back();
			numVisited++;
			if (idx >> 31) continue;
			idx &= 0x7FFFFFFF;
			auto node = nodes[idx];
			stack.push_back(node.leftIdx);
			stack.push_back(node.rightIdx);
			maxSize = std::max(maxSize, (int)stack.size());
		}
		if (numVisited != numObjs * 2 - 1)
			printf("Error: Not all nodes are accessible from the root. Only %d/%d nodes are found.\n", numVisited, 2 * numObjs - 1);
		if (maxStackSize != maxSize)
			printf("Error: Max stack size mismatch. Stored %d, found %d\n", maxStackSize, maxSize);

		// Check merging of indices and aabbs
		lbvhCheckIndexMerge(nodes, 0, numObjs);
		testAABBMatch(lbvhCheckAABBMerge(nodes, 0u), rootBounds, 0);

		// printf("Num nodes: %d\n", nodes.size());
		printf("Max stack size: %d\n", maxStackSize);
		printf("Node size: %d\n", sizeof(LBVHNode<float>));
		printf("LBVH self check complete.\n\n");
	}

	// Tests the LBVH
	void testLBVH() {
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

		// Create CUDA events for timing
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		float milliseconds = 0;

		// Build BVH
		printf("Building LBVH...\n");
		cudaEventRecord(start);
		LBVH bvh;
		bvh.compute(thrust::raw_pointer_cast(d_points.data()), N);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("LBVH build time: %.3f ms\n", milliseconds);

		// Query BVH
		printf("Querying LBVH...\n");
		cudaEventRecord(start);
		int numCols = bvh.query(thrust::raw_pointer_cast(d_res.data()), d_res.size());
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("LBVH query time: %.3f ms\n", milliseconds);

		// Print results
		printf("Getting results...\n");
		thrust::host_vector<int2> res(d_res.begin(), d_res.begin() + numCols);

		printf("%d collision pairs found on GPU.\n", res.size());

		// Brute force compute the same result using TBB
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
		
		// Time CPU computation
		auto cpu_start = std::chrono::high_resolution_clock::now();
		
		// Use TBB parallel_for for collision detection
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

		// Clean up CUDA events
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

#pragma endregion

}