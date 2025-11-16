#include "radix_sort.cuh"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <cuda_runtime.h>

namespace gpu_bvh {

void sortMortonCodesGPU(u_int32_t* d_codes, u_int32_t* d_indices, int N) {
    thrust::device_ptr<u_int32_t> dev_codes_ptr(d_codes);
    thrust::device_ptr<u_int32_t> dev_indices_ptr(d_indices);

    // Sort the morton codes and reorder the indices accordingly
    thrust::sort_by_key(
        thrust::device,
        dev_codes_ptr,
        dev_codes_ptr + N,
        dev_indices_ptr
    );
}

} // namespace gpu_bvh
