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

// Weighted graph structure for PARIS algorithm
struct WeightedGraph {
    std::unordered_map<uint32_t, std::unordered_map<uint32_t, double>> adjacency;
    std::unordered_map<uint32_t, double> node_weights;
    std::unordered_map<uint32_t, uint32_t> cluster_sizes;
    double total_weight = 0.0;

    // Add a node
    void add_node(uint32_t node) {
        if (adjacency.find(node) == adjacency.end()) {
            adjacency[node] = std::unordered_map<uint32_t, double>();
            node_weights[node] = 0.0;
            cluster_sizes[node] = 1;
        }
    }

    // Add an edge with weight
    void add_edge(uint32_t u, uint32_t v, double weight = 1.0) {
        adjacency[u][v] = weight;
    }

    // Check if edge exists
    bool has_edge(uint32_t u, uint32_t v) const {
        auto it = adjacency.find(u);
        if (it == adjacency.end()) return false;
        return it->second.find(v) != it->second.end();
    }

    // Get edge weight
    double get_edge_weight(uint32_t u, uint32_t v) const {
        auto it = adjacency.find(u);
        if (it == adjacency.end()) return 0.0;
        auto it2 = it->second.find(v);
        if (it2 == it->second.end()) return 0.0;
        return it2->second;
    }

    // Get neighbors of a node
    std::vector<uint32_t> get_neighbors(uint32_t node) const {
        std::vector<uint32_t> neighbors;
        auto it = adjacency.find(node);
        if (it != adjacency.end()) {
            for (const auto& [neighbor, weight] : it->second) {
                neighbors.push_back(neighbor);
            }
        }
        return neighbors;
    }

    // Get all nodes
    std::vector<uint32_t> get_nodes() const {
        std::vector<uint32_t> nodes;
        for (const auto& [node, _] : adjacency) {
            nodes.push_back(node);
        }
        return nodes;
    }

    // Remove a node
    void remove_node(uint32_t node) {
        adjacency.erase(node);
        node_weights.erase(node);
        cluster_sizes.erase(node);

        // Remove edges to this node from other nodes
        for (auto& [n, neighbors] : adjacency) {
            neighbors.erase(node);
        }
    }

    // Get number of nodes
    size_t num_nodes() const {
        return adjacency.size();
    }
};

// Convert unweighted Graph to WeightedGraph
WeightedGraph convert_to_weighted_graph(const Graph& g) {
    WeightedGraph wg;

    // Add all nodes
    for (uint32_t u = 0; u < g.num_nodes; ++u) {
        wg.add_node(u);
    }

    // Add all edges with default weight of 1.0
    // Only add each edge once (undirected graph)
    for (uint32_t u = 0; u < g.num_nodes; ++u) {
        for (uint32_t idx = g.row_ptr[u]; idx < g.row_ptr[u + 1]; ++idx) {
            uint32_t v = g.col_idx[idx];
            if (u <= v) {  // Only add edge once for undirected graph
                wg.add_edge(u, v, 1.0);
                wg.add_edge(v, u, 1.0);
            }
        }
    }

    // Calculate node weights and total weight
    wg.total_weight = 0.0;
    for (uint32_t u = 0; u < g.num_nodes; ++u) {
        double weight = 0.0;
        for (uint32_t idx = g.row_ptr[u]; idx < g.row_ptr[u + 1]; ++idx) {
            uint32_t v = g.col_idx[idx];
            double edge_weight = wg.get_edge_weight(u, v);
            weight += edge_weight;
            wg.total_weight += edge_weight;
            if (u != v) {
                wg.total_weight += edge_weight;
            }
        }
        wg.node_weights[u] = weight;
    }

    return wg;
}

// Reorder dendrogram
std::vector<DendrogramNode> reorder_dendrogram(const std::vector<DendrogramNode>& D) {
    size_t n = D.size() + 1;

    // Create ordering based on distance
    std::vector<std::pair<double, size_t>> order(n - 1);
    for (size_t i = 0; i < n - 1; ++i) {
        order[i] = {D[i].distance, i};
    }

    // Sort by distance
    std::sort(order.begin(), order.end());

    // Create index mapping
    std::unordered_map<uint32_t, uint32_t> node_index;
    for (uint32_t i = 0; i < n; ++i) {
        node_index[i] = i;
    }

    for (size_t t = 0; t < n - 1; ++t) {
        size_t orig_idx = order[t].second;
        node_index[n + orig_idx] = n + t;
    }

    // Reorder dendrogram
    std::vector<DendrogramNode> reordered(n - 1);
    for (size_t t = 0; t < n - 1; ++t) {
        size_t orig_idx = order[t].second;
        reordered[t] = {
            node_index[D[orig_idx].cluster_a],
            node_index[D[orig_idx].cluster_b],
            D[orig_idx].distance,
            D[orig_idx].size
        };
    }

    return reordered;
}

// PARIS algorithm
std::vector<DendrogramNode> paris(const Graph& g, bool verbose = false) {
    // Convert to weighted graph
    WeightedGraph F = convert_to_weighted_graph(g);
    uint32_t n = F.num_nodes();

    if (verbose) {
        std::cout << "Running PARIS algorithm on graph with " << n << " nodes" << std::endl;
        std::cout << "Total weight: " << F.total_weight << std::endl;
    }

    // Dendrogram
    std::vector<DendrogramNode> dendrogram;

    // Connected components
    std::vector<std::pair<uint32_t, uint32_t>> connected_components;

    // Cluster index
    uint32_t cluster_idx = n;

    while (F.num_nodes() > 0) {
        // Start nearest-neighbor chain
        std::vector<uint32_t> chain;
        auto nodes = F.get_nodes();
        if (nodes.empty()) break;

        chain.push_back(nodes[0]);

        while (!chain.empty()) {
            uint32_t a = chain.back();
            chain.pop_back();

            // Find nearest neighbor
            double dmin = std::numeric_limits<double>::infinity();
            int32_t b = -1;

            auto neighbors = F.get_neighbors(a);
            for (uint32_t v : neighbors) {
                if (v != a) {
                    double edge_weight = F.get_edge_weight(a, v);
                    double d = F.node_weights[v] * F.node_weights[a] / edge_weight / F.total_weight;

                    if (d < dmin) {
                        b = v;
                        dmin = d;
                    } else if (d == dmin) {
                        b = std::min(b, static_cast<int32_t>(v));
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

                    // Get neighbors before removing nodes
                    auto neighbors_a = F.get_neighbors(a);
                    auto neighbors_b = F.get_neighbors(b);

                    // Add edges from new cluster to neighbors of a
                    for (uint32_t v : neighbors_a) {
                        double weight = F.get_edge_weight(a, v);
                        F.add_edge(cluster_idx, v, weight);
                        F.add_edge(v, cluster_idx, weight);
                    }

                    // Add edges from new cluster to neighbors of b
                    for (uint32_t v : neighbors_b) {
                        double weight_b = F.get_edge_weight(b, v);
                        if (F.has_edge(cluster_idx, v)) {
                            double existing_weight = F.get_edge_weight(cluster_idx, v);
                            F.add_edge(cluster_idx, v, existing_weight + weight_b);
                            F.add_edge(v, cluster_idx, existing_weight + weight_b);
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
