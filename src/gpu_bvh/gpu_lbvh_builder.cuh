#ifndef GPU_LBVH_BUILDER_CUH
#define GPU_LBVH_BUILDER_CUH

#include <vector>
#include <stdint.h>
#include <cuda_runtime.h>

#include "gpu_bvh/lbvh_node.cuh"
#include "gpu_bvh/morton.cuh"
#include "gpu_bvh/radix_sort.cuh"

namespace gpu_bvh {

class LBVHBuilderGPU {
public:
    LBVHBuilderGPU(
        const float3* h_centers,
        const float3* h_bbox_min,
        const float3* h_bbox_max,
        int N
    );
    ~LBVHBuilderGPU();

    void build();

    void downloadNodes(std::vector<GPULBVHNode>& out_nodes) const;
    void downloadIndices(std::vector<uint32_t>& out_indices) const;

    int primitiveCount() const { return m_N; }
    int nodeCount() const { return m_totalNodes; }

private:
    const float3* m_h_centers;
    const float3* m_h_bbox_min;
    const float3* m_h_bbox_max;

    float3*   m_d_centers   = nullptr;
    float3*   m_d_bbox_min  = nullptr;
    float3*   m_d_bbox_max  = nullptr;
    uint32_t* m_d_morton    = nullptr;
    uint32_t* m_d_indices   = nullptr;
    GPULBVHNode* m_d_nodes = nullptr;

    int m_N = 0;
    int m_totalNodes = 0;

    void uploadPrimitives();
    void allocateDeviceBuffers();
    void freeDeviceBuffers();

    void buildTopologyOnGPU();
    void buildAABBsOnGPU();
};

} // namespace gpu_bvh

#endif // GPU_LBVH_BUILDER_CUH