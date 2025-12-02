#ifndef PARIS_H
#define PARIS_H

#include <vector>
#include <unordered_map>
#include <limits>
#include <algorithm>
#include <cmath>
#include <iostream>
#include "../data_structures/graph.h"

// Dendrogram entry: [cluster_a, cluster_b, distance, size]
struct DendrogramNode {
    uint32_t cluster_a;
    uint32_t cluster_b;
    double distance;
    uint32_t size;
};

// Optimized weighted graph structure using vectors for better cache locality
struct WeightedGraph {
    // Map from node ID to adjacency list (neighbor -> weight)
    std::unordered_map<uint32_t, std::unordered_map<uint32_t, double>> adjacency;

    // Use vectors for dense node properties (better cache locality)
    std::vector<double> node_weights;
    std::vector<uint32_t> cluster_sizes;

    // Active nodes set for quick existence checks
    std::unordered_map<uint32_t, bool> active_nodes;

    double total_weight = 0.0;
    uint32_t max_node_id = 0;

    void ensure_capacity(uint32_t node_id) {
        if (node_id >= max_node_id) {
            max_node_id = node_id + 1;
            if (node_weights.size() <= node_id) {
                node_weights.resize(node_id + 1024, 0.0);
                cluster_sizes.resize(node_id + 1024, 0);
            }
        }
    }

    void add_node(uint32_t node) {
        ensure_capacity(node);
        if (active_nodes.find(node) == active_nodes.end()) {
            adjacency[node] = std::unordered_map<uint32_t, double>();
            active_nodes[node] = true;
            if (cluster_sizes[node] == 0) {
                cluster_sizes[node] = 1;
            }
        }
    }

    void add_edge(uint32_t u, uint32_t v, double weight) {
        adjacency[u][v] = weight;
    }

    inline bool has_edge(uint32_t u, uint32_t v) const {
        auto it = adjacency.find(u);
        if (it == adjacency.end()) return false;
        return it->second.find(v) != it->second.end();
    }

    inline double get_edge_weight(uint32_t u, uint32_t v) const {
        auto it = adjacency.find(u);
        if (it == adjacency.end()) return 0.0;
        auto it2 = it->second.find(v);
        if (it2 == it->second.end()) return 0.0;
        return it2->second;
    }

    // Return const reference to avoid allocation
    inline const std::unordered_map<uint32_t, double>& get_neighbors_map(uint32_t node) const {
        static const std::unordered_map<uint32_t, double> empty;
        auto it = adjacency.find(node);
        if (it != adjacency.end()) {
            return it->second;
        }
        return empty;
    }

    void get_active_nodes(std::vector<uint32_t>& nodes) const {
        nodes.clear();
        nodes.reserve(active_nodes.size());
        for (const auto& [node, _] : active_nodes) {
            nodes.push_back(node);
        }
    }

    void remove_node(uint32_t node) {
        // Get neighbors before erasing
        auto it = adjacency.find(node);
        if (it != adjacency.end()) {
            // Remove back-edges efficiently
            for (const auto& [neighbor, _] : it->second) {
                auto neighbor_it = adjacency.find(neighbor);
                if (neighbor_it != adjacency.end()) {
                    neighbor_it->second.erase(node);
                }
            }
            adjacency.erase(it);
        }

        active_nodes.erase(node);
        // Keep vectors allocated but mark as inactive
        node_weights[node] = 0.0;
    }

    inline size_t num_nodes() const {
        return active_nodes.size();
    }
};

// Optimized single-pass conversion
WeightedGraph convert_to_weighted_graph(const Graph& g) {
    WeightedGraph wg;

    // Pre-allocate vectors
    wg.node_weights.resize(g.num_nodes, 0.0);
    wg.cluster_sizes.resize(g.num_nodes, 1);
    wg.max_node_id = g.num_nodes;
    wg.total_weight = 0.0;

    // Reserve space for adjacency map
    wg.adjacency.reserve(g.num_nodes);
    wg.active_nodes.reserve(g.num_nodes);

    // Single pass: add nodes, edges, and calculate weights
    for (uint32_t u = 0; u < g.num_nodes; ++u) {
        wg.adjacency[u] = std::unordered_map<uint32_t, double>();
        wg.active_nodes[u] = true;

        uint32_t degree = g.row_ptr[u + 1] - g.row_ptr[u];
        if (degree > 0) {
            wg.adjacency[u].reserve(degree);
        }

        double node_weight = 0.0;
        for (uint32_t idx = g.row_ptr[u]; idx < g.row_ptr[u + 1]; ++idx) {
            uint32_t v = g.col_idx[idx];
            wg.adjacency[u][v] = 1.0;
            node_weight += 1.0;
        }

        wg.node_weights[u] = node_weight;
        wg.total_weight += node_weight;
    }

    return wg;
}

// Reorder dendrogram
std::vector<DendrogramNode> reorder_dendrogram(const std::vector<DendrogramNode>& D) {
    if (D.empty()) return D;

    size_t n = D.size() + 1;

    // Create ordering based on distance
    std::vector<std::pair<double, size_t>> order;
    order.reserve(n - 1);
    for (size_t i = 0; i < n - 1; ++i) {
        order.emplace_back(D[i].distance, i);
    }

    // Sort by distance
    std::sort(order.begin(), order.end());

    // Create index mapping using vector for original nodes (faster than unordered_map)
    std::vector<uint32_t> node_index(n + n - 1);
    for (uint32_t i = 0; i < n; ++i) {
        node_index[i] = i;
    }

    for (size_t t = 0; t < n - 1; ++t) {
        size_t orig_idx = order[t].second;
        node_index[n + orig_idx] = n + t;
    }

    // Reorder dendrogram
    std::vector<DendrogramNode> reordered;
    reordered.reserve(n - 1);
    for (size_t t = 0; t < n - 1; ++t) {
        size_t orig_idx = order[t].second;
        reordered.push_back({
            node_index[D[orig_idx].cluster_a],
            node_index[D[orig_idx].cluster_b],
            D[orig_idx].distance,
            D[orig_idx].size
        });
    }

    return reordered;
}

// Optimized PARIS algorithm
std::vector<DendrogramNode> paris(const Graph& g, bool verbose = false) {
    // Convert to weighted graph
    WeightedGraph F = convert_to_weighted_graph(g);
    uint32_t n = F.num_nodes();

    if (verbose) {
        std::cout << "Running PARIS algorithm on graph with " << n << " nodes" << std::endl;
        std::cout << "Total weight: " << F.total_weight << std::endl;
    }

    // Pre-allocate dendrogram
    std::vector<DendrogramNode> dendrogram;
    dendrogram.reserve(n - 1);

    // Connected components
    std::vector<std::pair<uint32_t, uint32_t>> connected_components;

    // Cluster index
    uint32_t cluster_idx = n;

    // Reusable vectors to avoid repeated allocations
    std::vector<uint32_t> active_nodes;
    std::vector<uint32_t> chain;
    chain.reserve(1024);

    while (F.num_nodes() > 0) {
        // Start nearest-neighbor chain
        chain.clear();
        F.get_active_nodes(active_nodes);
        if (active_nodes.empty()) break;

        chain.push_back(active_nodes[0]);

        while (!chain.empty()) {
            uint32_t a = chain.back();
            chain.pop_back();

            // Find nearest neighbor - use const reference to avoid allocation
            double dmin = std::numeric_limits<double>::infinity();
            int32_t b = -1;

            const auto& neighbors = F.get_neighbors_map(a);
            double node_weight_a = F.node_weights[a];
            double inv_total_weight = 1.0 / F.total_weight;

            for (const auto& [v, edge_weight] : neighbors) {
                if (v != a) {
                    // Optimized distance calculation
                    double d = F.node_weights[v] * node_weight_a / edge_weight * inv_total_weight;

                    if (d < dmin) {
                        b = v;
                        dmin = d;
                    } else if (d == dmin && static_cast<int32_t>(v) < b) {
                        b = v;
                    }
                }
            }

            double d = dmin;

            if (!chain.empty()) {
                uint32_t c = chain.back();
                chain.pop_back();

                if (b == static_cast<int32_t>(c)) {
                    // Merge a and b
                    uint32_t size_a = F.cluster_sizes[a];
                    uint32_t size_b = F.cluster_sizes[b];

                    dendrogram.push_back({a, static_cast<uint32_t>(b), d, size_a + size_b});

                    if (verbose && dendrogram.size() % 1000 == 0) {
                        std::cout << "Merged " << dendrogram.size() << " clusters" << std::endl;
                    }

                    // Update graph - add new cluster node
                    F.add_node(cluster_idx);

                    // Copy neighbor lists before modification (references would be invalidated)
                    auto neighbors_a_copy = F.adjacency[a];
                    auto neighbors_b_copy = F.adjacency[b];

                    // Reserve space for new adjacency list
                    size_t total_neighbors = neighbors_a_copy.size() + neighbors_b_copy.size();
                    F.adjacency[cluster_idx].reserve(total_neighbors);

                    // Add edges from new cluster to neighbors of a
                    for (const auto& [v, weight] : neighbors_a_copy) {
                        F.add_edge(cluster_idx, v, weight);
                        F.add_edge(v, cluster_idx, weight);
                    }

                    // Add edges from new cluster to neighbors of b
                    for (const auto& [v, weight_b] : neighbors_b_copy) {
                        if (F.has_edge(cluster_idx, v)) {
                            double existing_weight = F.get_edge_weight(cluster_idx, v);
                            double new_weight = existing_weight + weight_b;
                            F.add_edge(cluster_idx, v, new_weight);
                            F.add_edge(v, cluster_idx, new_weight);
                        } else {
                            F.add_edge(cluster_idx, v, weight_b);
                            F.add_edge(v, cluster_idx, weight_b);
                        }
                    }

                    // Update node weight and size
                    F.node_weights[cluster_idx] = F.node_weights[a] + F.node_weights[b];
                    F.cluster_sizes[cluster_idx] = size_a + size_b;

                    // Remove old nodes
                    F.remove_node(a);
                    F.remove_node(b);

                    // Increment cluster index
                    cluster_idx++;
                } else {
                    chain.push_back(c);
                    chain.push_back(a);
                    chain.push_back(b);
                }
            } else if (b >= 0) {
                chain.push_back(a);
                chain.push_back(b);
            } else {
                // Isolated node - add to connected components
                connected_components.push_back({a, F.cluster_sizes[a]});
                F.remove_node(a);
            }
        }
    }

    // Add connected components to dendrogram
    if (!connected_components.empty()) {
        auto [a, size_a] = connected_components.back();
        connected_components.pop_back();

        uint32_t total_size = size_a;
        for (const auto& [b, size_b] : connected_components) {
            total_size += size_b;
            dendrogram.push_back({a, b, std::numeric_limits<double>::infinity(), total_size});
            a = cluster_idx;
            cluster_idx++;
        }
    }

    if (verbose) {
        std::cout << "Created dendrogram with " << dendrogram.size() << " merges" << std::endl;
    }

    // Reorder dendrogram
    return reorder_dendrogram(dendrogram);
}

#endif // PARIS_H
