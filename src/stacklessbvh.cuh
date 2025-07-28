#pragma once
// Xiao Cheng, 2025
#include "bound.h"
#include "thrust/device_vector.h"
#include "thrust/device_ptr.h"
#include <memory>
#include "typedef.h"

namespace culbvh {
    using aabb = Bound<float>;
    struct __align__(16) stacklessnode {
        int lc;
        int escape;
        aabb bound;
    };

    class LBVHStackless {
	public:
        
        using vec_type = float3;

        LBVHStackless();
        ~LBVHStackless() ;

        bool is_valid() const { return numObjs > 0; }

        size_t size() const { return numObjs; }

        void compute(aabb* devicePtr, size_t size);

        size_t query();

		size_t queryOther(aabb* devicePtr, size_t size);

        int type = 1; // 0: quant node 16 bytes,   1: 32 bytes 

    private:
        struct thrustImpl;
        std::unique_ptr<thrustImpl> impl;
        aabb rootBounds;
        size_t numObjs{ 0 };

        //cub storage
        size_t temp_storage_bytes = 0;
        size_t max_storage_bytes = 0;
        void* d_temp_storage = nullptr;


        //result 
        int max_cpNum = 5000000;
        int h_cpNum;
        int* d_cpNum;
        int2* d_cpRes;
		bool allocated{ false };

        unsigned int* d_queryMtCode;
        aabb* d_querySceneBox;
		int* d_querySortedId;
        int queryNum = 0;

		

    };

	void testStacklessLBVH();
}