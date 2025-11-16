#include "morton.cuh"
#include <cuda_runtime.h>
#include <algorithm>

namespace gpu_bvh {

__host__ __device__ uint32_t expandBits(uint32_t v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

__host__ __device__ uint32_t morton3D(float x, float y, float z) {
    x = fminf(fmaxf(x * 1024.0f, 0.0f), 1023.0f);
    y = fminf(fmaxf(y * 1024.0f, 0.0f), 1023.0f);
    z = fminf(fmaxf(z * 1024.0f, 0.0f), 1023.0f);

    uint32_t xx = expandBits((uint32_t)x);
    uint32_t yy = expandBits((uint32_t)y);
    uint32_t zz = expandBits((uint32_t)z);

    return (xx << 2) | (yy << 1) | zz;
}

__global__ void mortonKernel(const float3* points, uint32_t* codes, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float3 p = points[idx];
    codes[idx] = morton3D(p.x, p.y, p.z);
}

void computeMortonCodesGPU(const float3* d_points, uint32_t* d_codes, int N) {
    int block = 256;
    int grid = (N + block - 1) / block;
    
    mortonKernel<<<grid, block>>>(d_points, d_codes, N);
    cudaDeviceSynchronize();
}

} // namespace gpu_bvh
