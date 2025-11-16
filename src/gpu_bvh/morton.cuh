#ifndef MORTON_CUH
#define MORTON_CUH

#include <cuda_runtime.h>
#include <stdint.h>

namespace gpu_bvh {

__host__ __device__ uint32_t expandBits(uint32_t v);
__host__ __device__ uint32_t morton3D(float x, float y, float z);

void computeMortonCodesGPU(const float3* d_points, uint32_t* d_codes, int N);

} // namespace gpu_bvh

#endif // MORTON_CUH