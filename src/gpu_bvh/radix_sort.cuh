#ifndef RADIX_SORT_CUH
#define RADIX_SORT_CUH

#include <cuda_runtime.h>
#include <stdint.h>

namespace gpu_bvh {

void sortMortonCodesGPU(u_int32_t* d_codes, u_int32_t* d_indices, int N);

} // namespace gpu_bvh

#endif // RADIX_SORT_CUH