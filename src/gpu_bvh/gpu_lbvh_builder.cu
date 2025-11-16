#include "gpu_bvh/gpu_lbvh_builder.cuh"
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <cstdio>

namespace gpu_bvh {

// Helpers

static inline void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__device__ inline int clz_prefix(const uint32_t* morton, int i, int j, int N) {
    if (j < 0 || j >= N) return -1;
    
    if (morton[i] == morton[j]) return 32 + __clz(i ^ j);
    return __clz(morton[i] ^ morton[j]);
}

// Constructor and Destructor

LBVHBuilderGPU::LBVHBuilderGPU(
    const float3* h_centers,
    const float3* h_bbox_min,
    const float3* h_bbox_max,
    int N
) :
    m_h_centers(h_centers),
    m_h_bbox_min(h_bbox_min),
    m_h_bbox_max(h_bbox_max),
    m_N(N)
{
    m_totalNodes = 2 * N - 1;
    allocateDeviceBuffers();
    uploadPrimitives();
}

LBVHBuilderGPU::~LBVHBuilderGPU() {
    freeDeviceBuffers();
}

// memory management

void LBVHBuilderGPU::allocateDeviceBuffers() {
    if (m_N <= 0) return;

    checkCuda(cudaMalloc(&m_d_centers,  m_N * sizeof(float3)), "Allocating device centers");
    checkCuda(cudaMalloc(&m_d_bbox_min, m_N * sizeof(float3)), "Allocating device bbox_min");
    checkCuda(cudaMalloc(&m_d_bbox_max, m_N * sizeof(float3)), "Allocating device bbox_max");
    checkCuda(cudaMalloc(&m_d_morton,   m_N * sizeof(uint32_t)), "Allocating device morton codes");
    checkCuda(cudaMalloc(&m_d_indices,  m_N * sizeof(uint32_t)), "Allocating device indices");
    checkCuda(cudaMalloc(&m_d_nodes,    m_totalNodes * sizeof(GPULBVHNode)), "Allocating device LBVH nodes");
}

void LBVHBuilderGPU::uploadPrimitives() {
    if (m_N <= 0) return;

    checkCuda(cudaMemcpy(m_d_centers,  m_h_centers,  m_N * sizeof(float3), cudaMemcpyHostToDevice), "Uploading centers");
    checkCuda(cudaMemcpy(m_d_bbox_min, m_h_bbox_min, m_N * sizeof(float3), cudaMemcpyHostToDevice), "Uploading bbox_min");
    checkCuda(cudaMemcpy(m_d_bbox_max, m_h_bbox_max, m_N * sizeof(float3), cudaMemcpyHostToDevice), "Uploading bbox_max");
}

void LBVHBuilderGPU::freeDeviceBuffers() {
    if (m_d_centers)  cudaFree(m_d_centers);
    if (m_d_bbox_min) cudaFree(m_d_bbox_min);
    if (m_d_bbox_max) cudaFree(m_d_bbox_max);
    if (m_d_morton)   cudaFree(m_d_morton);
    if (m_d_indices)  cudaFree(m_d_indices);
    if (m_d_nodes)    cudaFree(m_d_nodes);

    m_d_centers  = nullptr;
    m_d_bbox_min = nullptr;
    m_d_bbox_max = nullptr;
    m_d_morton   = nullptr;
    m_d_indices  = nullptr;
    m_d_nodes    = nullptr;
}


// Build LBVH

void LBVHBuilderGPU::build() {
    if (m_N <= 0) return;

    {
        thrust::device_ptr<uint32_t> idx_ptr(m_d_indices);
        thrust::sequence(idx_ptr, idx_ptr + m_N);
    }

    computeMortonCodesGPU(m_d_centers, m_d_morton, m_N);
    sortMortonCodesGPU(m_d_morton, m_d_indices, m_N);

    buildTopologyOnGPU();
    buildAABBsOnGPU();
}

// Download results

void LBVHBuilderGPU::downloadNodes(std::vector<GPULBVHNode>& out_nodes) const {
    out_nodes.resize(m_totalNodes);
    if (m_totalNodes == 0) return;

    checkCuda(cudaMemcpy(out_nodes.data(), m_d_nodes, m_totalNodes * sizeof(GPULBVHNode), cudaMemcpyDeviceToHost), "Downloading LBVH nodes");
}

void LBVHBuilderGPU::downloadIndices(std::vector<uint32_t>& out_indices) const {
    out_indices.resize(m_N);
    if (m_N == 0) return;

    checkCuda(cudaMemcpy(out_indices.data(), m_d_indices, m_N * sizeof(uint32_t), cudaMemcpyDeviceToHost), "Downloading LBVH indices");
}

// Construct Topology (Karras 2012)

__global__ void initLeafNodesKernel(GPULBVHNode* nodes, const uint32_t* indicies, int N) {
    int leafIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (leafIdx >= N) return;

    int nodeIdx = (N - 1) + leafIdx;
    GPULBVHNode& node = nodes[nodeIdx];

    node.left    = -1;
    node.right   = -1;
    node.parent  = -1;
    node.is_leaf = true;
    node.prim_index = (int)indicies[leafIdx];

    node.bbox_min = make_float3(0.0f, 0.0f, 0.0f);
    node.bbox_max = make_float3(0.0f, 0.0f, 0.0f);
}

__global__ void buildInternalNodesKernel(GPULBVHNode* nodes, const uint32_t* morton, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N - 1) return;

    int delta_next = clz_prefix(morton, i, i + 1, N);
    int delta_prev = clz_prefix(morton, i, i - 1, N);

    int d = (delta_next > delta_prev) ? 1 : -1;

    int delta_min = clz_prefix(morton, i, i - d, N);

    int l_max = 2;
    while (true) {
        int j = i + l_max * d;
        if (j < 0 || j >= N) break;

        int delta = clz_prefix(morton, i, j, N);
        if (delta < delta_min) break;
        l_max <<= 1;
    }

    int l = 0;
    for (int t = l_max >> 1; t >= 1; t >>= 1) {
        int j = i + (l + t) * d;
        if (j < 0 || j >= N) continue;

        int delta = clz_prefix(morton, i, j, N);
        if (delta >= delta_min) l += t;
    }

    int j = i + l * d;
    int first = min(i, j);
    int last  = max(i, j);

    int delta_node = clz_prefix(morton, first, last, N);

    int split = first;
    int step = last - first;

    while (step > 1) {
        step = (step + 1) >> 1;
        int mid = split + step;

        if (mid < last) {
            int delta_mid = clz_prefix(morton, first, mid, N);
            if (delta_mid > delta_node) split = mid;
        }
    }

    int left_index;
    int right_index;

    int split_internal = split;
    if (split_internal == first) {
        left_index = (N - 1) + split_internal;
    } else {
        left_index = split_internal;
    }

    int split_next = split + 1;
    if (split_next == last) {
        right_index = (N - 1) + split_next;
    } else {
        right_index = split_next;
    }

    GPULBVHNode& node = nodes[i];
    node.left   = left_index;
    node.right  = right_index;
    node.parent = -1;
    node.is_leaf = false;
}

__global__ void setParentKernel(GPULBVHNode* nodes, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N - 1) return;

    GPULBVHNode& node = nodes[i];
    int L = node.left;
    int R = node.right;

    if (L >= 0) nodes[L].parent = i;
    if (R >= 0) nodes[R].parent = i;
}

void LBVHBuilderGPU::buildTopologyOnGPU() {
    if (m_N <= 0) return;

    // Initialize leaf nodes
    {
        int block = 128;
        int grid = (m_N + block - 1) / block;;
        initLeafNodesKernel<<<grid, block>>>(m_d_nodes, m_d_indices, m_N);
        checkCuda(cudaGetLastError(), "Launching initLeafNodesKernel");
    }

    // Build internal nodes
    {
        int internalCount = m_N - 1;
        int block = 128;
        int grid = (internalCount + block - 1) / block;
        buildInternalNodesKernel<<<grid, block>>>(m_d_nodes, m_d_morton, m_N);
        checkCuda(cudaGetLastError(), "Launching buildInternalNodesKernel");
    }

    // Set parent pointers
    {
        int internalCount = m_N - 1;
        int block = 128;
        int grid = (internalCount + block - 1) / block;
        setParentKernel<<<grid, block>>>(m_d_nodes, m_N);
        checkCuda(cudaGetLastError(), "Launching setParentKernel");
    }
}

// Construct AABBs

static inline float3 f3_min(const float3& a, const float3& b) {
    return make_float3(
        fminf(a.x, b.x),
        fminf(a.y, b.y),
        fminf(a.z, b.z)
    );
}

static inline float3 f3_max(const float3& a, const float3& b) {
    return make_float3(
        fmaxf(a.x, b.x),
        fmaxf(a.y, b.y),
        fmaxf(a.z, b.z)
    );
}

void LBVHBuilderGPU::buildAABBsOnGPU() {
    if (m_N <= 0) return;

    std::vector<GPULBVHNode> h_nodes(m_totalNodes);
    checkCuda(cudaMemcpy(h_nodes.data(), m_d_nodes, m_totalNodes * sizeof(GPULBVHNode), cudaMemcpyDeviceToHost), "Downloading nodes for AABB construction");

    for (int leafIdx = 0; leafIdx < m_N; leafIdx++) {
        int nodeIdx = (m_N - 1) + leafIdx;
        GPULBVHNode& node = h_nodes[nodeIdx];

        int prim = node.prim_index;

        const float3 bbox_min = m_h_bbox_min[prim];
        const float3 bbox_max = m_h_bbox_max[prim];

        node.bbox_min = bbox_min;
        node.bbox_max = bbox_max;
    }

    for (int i = m_N - 2; i >= 0; i--) {
        GPULBVHNode& node = h_nodes[i];

        const GPULBVHNode& leftChild  = h_nodes[node.left];
        const GPULBVHNode& rightChild = h_nodes[node.right];

        node.bbox_min = f3_min(leftChild.bbox_min, rightChild.bbox_min);
        node.bbox_max = f3_max(leftChild.bbox_max, rightChild.bbox_max);
    }

    checkCuda(cudaMemcpy(m_d_nodes, h_nodes.data(), m_totalNodes * sizeof(GPULBVHNode), cudaMemcpyHostToDevice), "Uploading nodes after AABB construction");
}

} // namespace gpu_bvh
