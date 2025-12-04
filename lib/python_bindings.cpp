#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <map>
#include <set>
#include "data_structures/graph.h"
#include "data_structures/hierarchical_clustering.h"
#include "algorithm/champaign.h"
#include "algorithm/paris.h"
#include "algorithm/leiden.h"
#include "io/graph_io.h"

namespace py = pybind11;

// Python-friendly graph wrapper
class GraphWrapper {
public:
    Graph g;

    GraphWrapper() : g() {}

    void add_edge(uint32_t u, uint32_t v) {
        g.add_edge(u, v);
    }

    uint32_t num_nodes() const {
        return g.num_nodes;
    }

    uint32_t num_edges() const {
        return g.num_edges;
    }

    std::vector<uint32_t> neighbors(uint32_t node) const {
        std::vector<uint32_t> result;
        for (uint32_t neighbor : g.get_neighbors(node)) {
            result.push_back(neighbor);
        }
        return result;
    }

    // Get original node ID from internal ID
    uint64_t get_original_id(uint32_t internal_id) const {
        if (internal_id >= g.id_map.size()) {
            throw std::out_of_range("Internal ID out of range");
        }
        return g.id_map[internal_id];
    }

    // Get internal ID from original node ID
    uint32_t get_internal_id(uint64_t original_id) const {
        auto it = g.node_map.find(original_id);
        if (it == g.node_map.end()) {
            throw std::out_of_range("Original ID not found in graph");
        }
        return it->second;
    }

    // Get all original node IDs
    py::list get_original_ids() const {
        py::list result;
        for (uint64_t original_id : g.id_map) {
            result.append(original_id);
        }
        return result;
    }
};

// Python-friendly dendrogram wrapper
class DendrogramWrapper {
public:
    std::vector<DendrogramNode> nodes;
    uint32_t n_nodes;

    DendrogramWrapper() : nodes(), n_nodes(0) {}

    DendrogramWrapper(std::vector<DendrogramNode> dendro, uint32_t n)
        : nodes(dendro), n_nodes(n) {}

    size_t size() const {
        return nodes.size();
    }

    // Get merge information
    py::dict get_merge(size_t idx) const {
        if (idx >= nodes.size()) {
            throw std::out_of_range("Merge index out of range");
        }

        const auto& node = nodes[idx];
        py::dict merge;
        merge["left"] = node.cluster_a;
        merge["right"] = node.cluster_b;
        merge["distance"] = node.distance;
        merge["size"] = node.size;
        return merge;
    }

    // Get all merges as list of dicts
    py::list get_all_merges() const {
        py::list result;
        for (size_t i = 0; i < nodes.size(); i++) {
            result.append(get_merge(i));
        }
        return result;
    }

    // Leiden refinement
    py::dict leiden(const GraphWrapper& graph_wrapper, double gamma,
                   uint32_t max_iterations = 10,
                   uint32_t random_seed = 42,
                   bool verbose = false) const {
        Partition partition = leiden_from_dendrogram(
            graph_wrapper.g, nodes, gamma, max_iterations, random_seed, verbose
        );

        // Convert partition to Python format with ORIGINAL node IDs
        py::list clusters;
        for (const auto& [comm_id, members] : partition.community_members) {
            py::list cluster;
            for (uint32_t internal_node_id : members) {
                // Map back to original ID
                uint64_t original_id = graph_wrapper.g.id_map[internal_node_id];
                cluster.append(original_id);
            }
            clusters.append(cluster);
        }

        // Calculate CPM
        double cpm = calculate_partition_cpm(graph_wrapper.g, partition, gamma);

        py::dict result;
        result["clusters"] = clusters;
        result["cpm"] = cpm;
        result["num_clusters"] = partition.num_communities();
        result["gamma"] = gamma;

        return result;
    }

    // Extract clusters at a given distance threshold
    py::list extract_clusters(double max_distance) const {
        // Build union-find structure
        std::vector<uint32_t> parent(n_nodes);
        for (uint32_t i = 0; i < n_nodes; i++) {
            parent[i] = i;
        }

        // Process merges up to max_distance
        uint32_t cluster_id = n_nodes;
        for (const auto& merge : nodes) {
            if (merge.distance > max_distance) {
                break;
            }

            // Find roots
            uint32_t root_a = merge.cluster_a;
            while (root_a >= n_nodes && root_a - n_nodes < parent.size()) {
                root_a = parent[root_a];
            }

            uint32_t root_b = merge.cluster_b;
            while (root_b >= n_nodes && root_b - n_nodes < parent.size()) {
                root_b = parent[root_b];
            }

            // Union
            if (parent.size() <= cluster_id - n_nodes) {
                parent.resize(cluster_id - n_nodes + 1);
            }
            parent[cluster_id - n_nodes] = cluster_id;

            // Point old roots to new cluster
            if (root_a < n_nodes) {
                parent[root_a] = cluster_id;
            } else if (root_a - n_nodes < parent.size()) {
                parent[root_a - n_nodes] = cluster_id;
            }

            if (root_b < n_nodes) {
                parent[root_b] = cluster_id;
            } else if (root_b - n_nodes < parent.size()) {
                parent[root_b - n_nodes] = cluster_id;
            }

            cluster_id++;
        }

        // Find clusters
        std::map<uint32_t, std::vector<uint32_t>> clusters;
        for (uint32_t i = 0; i < n_nodes; i++) {
            uint32_t root = i;
            while (root < parent.size() && parent[root] != root && root < n_nodes + parent.size()) {
                root = parent[root];
            }
            clusters[root].push_back(i);
        }

        // Convert to Python list of lists
        py::list result;
        for (const auto& [root, members] : clusters) {
            py::list cluster;
            for (uint32_t node : members) {
                cluster.append(node);
            }
            if (py::len(cluster) > 0) {
                result.append(cluster);
            }
        }

        return result;
    }

    // Save dendrogram as JSON file
    void save_json(const std::string& filename,
                   const std::string& algorithm = "Champaign",
                   uint32_t num_edges = 0) const {
        // Build tree from dendrogram
        uint32_t root_id = n_nodes + nodes.size() - 1;
        TreeNode root = build_tree_recursive(root_id);

        // Write to file
        std::ofstream out(filename);
        if (!out.is_open()) {
            throw std::runtime_error("Could not open output file: " + filename);
        }

        out << "{\n";
        out << "  \"algorithm\": \"" << algorithm << "\",\n";
        out << "  \"num_nodes\": " << n_nodes << ",\n";
        out << "  \"num_edges\": " << num_edges << ",\n";
        out << "  \"hierarchy\": ";

        write_tree_as_json(out, root, 1);

        out << "\n}\n";
        out.close();
    }

private:
    // Tree node structure for JSON output
    struct TreeNode {
        uint32_t id;
        std::string type;
        double distance;
        uint32_t count;
        std::vector<TreeNode> children;
    };

    // Build tree recursively from dendrogram
    TreeNode build_tree_recursive(uint32_t node_id) const {
        TreeNode node;
        node.id = node_id;

        if (node_id < n_nodes) {
            // Leaf node
            node.type = "leaf";
            node.count = 1;
            node.distance = 0.0;
        } else {
            // Internal node - find the merge that created this cluster
            uint32_t merge_index = node_id - n_nodes;
            if (merge_index < nodes.size()) {
                const auto& merge = nodes[merge_index];
                node.type = "cluster";
                node.distance = merge.distance;
                node.count = merge.size;

                // Recursively build children
                node.children.push_back(build_tree_recursive(merge.cluster_a));
                node.children.push_back(build_tree_recursive(merge.cluster_b));
            }
        }

        return node;
    }

    // Write tree as JSON
    void write_tree_as_json(std::ostream& out, const TreeNode& node, int indent) const {
        std::string indent_str(indent * 2, ' ');
        std::string next_indent_str((indent + 1) * 2, ' ');

        out << indent_str << "{\n";
        out << next_indent_str << "\"id\": " << node.id << ",\n";
        out << next_indent_str << "\"type\": \"" << node.type << "\"";

        if (node.type == "leaf") {
            out << ",\n" << next_indent_str << "\"name\": \"" << node.id << "\",\n";
            out << next_indent_str << "\"count\": " << node.count << "\n";
        } else {
            out << ",\n" << next_indent_str << "\"distance\": ";
            if (std::isinf(node.distance)) {
                out << "null";
            } else {
                out << node.distance;
            }
            out << ",\n";
            out << next_indent_str << "\"count\": " << node.count << ",\n";
            out << next_indent_str << "\"children\": [\n";

            for (size_t i = 0; i < node.children.size(); ++i) {
                write_tree_as_json(out, node.children[i], indent + 2);
                if (i < node.children.size() - 1) {
                    out << ",\n";
                } else {
                    out << "\n";
                }
            }

            out << next_indent_str << "]\n";
        }

        out << indent_str << "}";
    }
};

// Load graph from TSV
GraphWrapper* load_from_tsv(const std::string& filename, bool verbose = false) {
    auto* wrapper = new GraphWrapper();

    // Use the optimized parallel loader from graph_io.h
    wrapper->g = load_undirected_tsv_edgelist_parallel(filename, 1, verbose);

    return wrapper;
}

// Run Champaign algorithm
DendrogramWrapper* run_champaign(const GraphWrapper& graph_wrapper, bool verbose = false) {
    std::vector<DendrogramNode> dendrogram = champaign(graph_wrapper.g, verbose);
    return new DendrogramWrapper(dendrogram, graph_wrapper.g.num_nodes);
}

// Run Paris algorithm
DendrogramWrapper* run_paris(const GraphWrapper& graph_wrapper, bool verbose = false) {
    std::vector<DendrogramNode> dendrogram = paris(graph_wrapper.g, verbose);
    return new DendrogramWrapper(dendrogram, graph_wrapper.g.num_nodes);
}

// Pybind11 module definition
PYBIND11_MODULE(champaign, m) {
    m.doc() = "Champaign and Paris hierarchical clustering algorithms";

    // Graph class
    py::class_<GraphWrapper>(m, "Graph")
        .def(py::init<>())
        .def("add_edge", &GraphWrapper::add_edge, "Add an edge to the graph",
             py::arg("u"), py::arg("v"))
        .def("num_nodes", &GraphWrapper::num_nodes, "Get number of nodes")
        .def("num_edges", &GraphWrapper::num_edges, "Get number of edges")
        .def("neighbors", &GraphWrapper::neighbors, "Get neighbors of a node",
             py::arg("node"))
        .def("get_original_id", &GraphWrapper::get_original_id, "Get original node ID from internal ID",
             py::arg("internal_id"))
        .def("get_internal_id", &GraphWrapper::get_internal_id, "Get internal ID from original node ID",
             py::arg("original_id"))
        .def("get_original_ids", &GraphWrapper::get_original_ids, "Get list of all original node IDs")
        .def("__repr__", [](const GraphWrapper& g) {
            return "Graph(nodes=" + std::to_string(g.num_nodes()) +
                   ", edges=" + std::to_string(g.num_edges()) + ")";
        });

    // Dendrogram class
    py::class_<DendrogramWrapper>(m, "Dendrogram")
        .def(py::init<>())
        .def("size", &DendrogramWrapper::size, "Number of merges in dendrogram")
        .def("get_merge", &DendrogramWrapper::get_merge, "Get merge information at index",
             py::arg("idx"))
        .def("get_all_merges", &DendrogramWrapper::get_all_merges,
             "Get all merges as list of dicts")
        .def("extract_clusters", &DendrogramWrapper::extract_clusters,
             "Extract clusters at given distance threshold",
             py::arg("max_distance"))
        .def("leiden", &DendrogramWrapper::leiden,
             "Apply Leiden refinement at resolution gamma",
             py::arg("graph"), py::arg("gamma"),
             py::arg("max_iterations") = 10,
             py::arg("random_seed") = 42,
             py::arg("verbose") = false)
        .def("save_json", &DendrogramWrapper::save_json,
             "Save dendrogram to JSON file",
             py::arg("filename"),
             py::arg("algorithm") = "Champaign",
             py::arg("num_edges") = 0)
        .def("__len__", &DendrogramWrapper::size)
        .def("__repr__", [](const DendrogramWrapper& d) {
            return "Dendrogram(merges=" + std::to_string(d.size()) +
                   ", nodes=" + std::to_string(d.n_nodes) + ")";
        });

    // Module functions
    m.def("from_tsv", &load_from_tsv, "Load graph from TSV file",
          py::arg("filename"), py::arg("verbose") = false,
          py::return_value_policy::take_ownership);

    m.def("champaign", &run_champaign, "Run Champaign algorithm",
          py::arg("graph"), py::arg("verbose") = false,
          py::return_value_policy::take_ownership);

    m.def("paris", &run_paris, "Run Paris algorithm",
          py::arg("graph"), py::arg("verbose") = false,
          py::return_value_policy::take_ownership);
}
