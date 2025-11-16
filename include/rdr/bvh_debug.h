#pragma once
#include <fstream>
#include <string>
#include "rdr/bvh_tree.h"

RDR_NAMESPACE_BEGIN

template <typename NodeType>
void dumpBVHTree(
    const BVHTree<NodeType>& tree,
    const std::string& filename)
{
    std::ofstream fout(filename);
    if (!fout.is_open()) return;

    const auto &internal_nodes = tree.getInternalNodes();
    const auto &root_index = tree.getRootIndex();

    fout << "==== BVH Dump ====\n";
    fout << "Node count = " << internal_nodes.size() << "\n";
    fout << "Root index = " << root_index << "\n\n";

    for (size_t i = 0; i < internal_nodes.size(); i++) {
        const auto& n = internal_nodes[i];
        fout << "Node[" << i << "] ";
        fout << (n.is_leaf ? "Leaf  " : "Internal  ");

        fout << "L=" << n.left_index
             << " R=" << n.right_index
             << " span=[" << n.span_left << "," << n.span_right << ") ";

        fout << "AABB: ("
             << n.aabb.low_bnd.x << ","
             << n.aabb.low_bnd.y << ","
             << n.aabb.low_bnd.z << ") -> ("
             << n.aabb.upper_bnd.x << ","
             << n.aabb.upper_bnd.y << ","
             << n.aabb.upper_bnd.z << ")\n";
    }
}

RDR_NAMESPACE_END