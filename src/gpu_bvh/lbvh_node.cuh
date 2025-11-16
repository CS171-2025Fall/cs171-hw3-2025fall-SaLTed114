#ifndef LBVH_NODE_CUH
#define LBVH_NODE_CUH

#include <cuda_runtime.h>

namespace gpu_bvh {

struct GPULBVHNode {
    int left;     // Index of the left child node
    int right;    // Index of the right child node
    int parent;   // Index of the parent node
    bool is_leaf; // Flag indicating if the node is a leaf

    int prim_index; // Index of the primitive (valid if leaf)

    float3 bbox_min; // Minimum corner of the bounding box
    float3 bbox_max; // Maximum corner of the bounding box
};

} // namespace gpu_bvh

#endif // LBVH_NODE_CUH