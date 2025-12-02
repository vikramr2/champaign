#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <unordered_map>
#include "../lib/data_structures/graph.h"
#include "../lib/io/graph_io.h"
#include "../lib/algorithm/paris.h"
#include "../lib/algorithm/champaign.h"

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <input_file> [options]" << std::endl;
    std::cout << "\nDescription:" << std::endl;
    std::cout << "  Run hierarchical clustering algorithm on a graph" << std::endl;
    std::cout << "\nOptions:" << std::endl;
    std::cout << "  -a <algorithm>    - Algorithm: champaign or paris (default: champaign)" << std::endl;
    std::cout << "  -o <output_file>  - Output file for results (default: print to stdout)" << std::endl;
    std::cout << "  -f <format>       - Output format: json or csv (default: json)" << std::endl;
    std::cout << "  -t <num_threads>  - Number of threads (default: hardware concurrency)" << std::endl;
    std::cout << "  -v                - Verbose output" << std::endl;
    std::cout << "\nAlgorithms:" << std::endl;
    std::cout << "  champaign  - Size-based distance metric: d = (n_a * n_b) / (total_weight * p(a,b))" << std::endl;
    std::cout << "  paris      - Degree-based distance metric: d = p(i) * p(j) / p(i,j) / total_weight" << std::endl;
}

std::string escape_json_string(const std::string& s) {
    std::ostringstream o;
    for (char c : s) {
        switch (c) {
            case '"': o << "\\\""; break;
            case '\\': o << "\\\\"; break;
            case '\b': o << "\\b"; break;
            case '\f': o << "\\f"; break;
            case '\n': o << "\\n"; break;
            case '\r': o << "\\r"; break;
            case '\t': o << "\\t"; break;
            default:
                if (c < 32) {
                    o << "\\u" << std::hex << std::setfill('0') << std::setw(4) << (int)c;
                } else {
                    o << c;
                }
        }
    }
    return o.str();
}

struct TreeNode {
    uint32_t id;
    std::string name;
    std::string type;
    double distance;
    uint32_t count;
    std::vector<TreeNode> children;
};

TreeNode build_tree_from_dendrogram(const std::vector<DendrogramNode>& dendrogram,
                                     uint32_t node_id,
                                     uint32_t num_original_nodes) {
    TreeNode node;
    node.id = node_id;

    if (node_id < num_original_nodes) {
        // Leaf node
        node.name = std::to_string(node_id);
        node.type = "leaf";
        node.count = 1;
        node.distance = 0.0;
    } else {
        // Internal node - find the merge that created this cluster
        uint32_t merge_index = node_id - num_original_nodes;
        if (merge_index < dendrogram.size()) {
            const auto& merge = dendrogram[merge_index];
            node.type = "cluster";
            node.distance = merge.distance;
            node.count = merge.size;

            // Recursively build children
            node.children.push_back(build_tree_from_dendrogram(dendrogram, merge.cluster_a, num_original_nodes));
            node.children.push_back(build_tree_from_dendrogram(dendrogram, merge.cluster_b, num_original_nodes));
        }
    }

    return node;
}

void write_tree_as_json(std::ostream& out, const TreeNode& node, int indent = 0) {
    std::string indent_str(indent * 2, ' ');
    std::string next_indent_str((indent + 1) * 2, ' ');

    out << indent_str << "{\n";
    out << next_indent_str << "\"id\": " << node.id << ",\n";
    out << next_indent_str << "\"type\": \"" << node.type << "\",\n";

    if (node.type == "leaf") {
        out << next_indent_str << "\"name\": \"" << escape_json_string(node.name) << "\",\n";
        out << next_indent_str << "\"count\": " << node.count << "\n";
    } else {
        out << next_indent_str << "\"distance\": ";
        if (std::isinf(node.distance)) {
            out << "Infinity";
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

void save_dendrogram_json(const std::vector<DendrogramNode>& dendrogram,
                          uint32_t num_nodes,
                          uint32_t num_edges,
                          const std::string& output_file,
                          const std::string& algorithm = "Champaign") {
    // Build the tree from the dendrogram
    // The root is the last merge
    uint32_t root_id = num_nodes + dendrogram.size() - 1;
    TreeNode root = build_tree_from_dendrogram(dendrogram, root_id, num_nodes);

    std::ofstream out(output_file);
    if (!out.is_open()) {
        throw std::runtime_error("Could not open output file: " + output_file);
    }

    out << "{\n";
    out << "  \"algorithm\": \"" << algorithm << "\",\n";
    out << "  \"num_nodes\": " << num_nodes << ",\n";
    out << "  \"num_edges\": " << num_edges << ",\n";
    out << "  \"hierarchy\": ";

    write_tree_as_json(out, root, 1);

    out << "\n}\n";
    out.close();
}

void save_dendrogram_csv(const std::vector<DendrogramNode>& dendrogram,
                         const std::string& output_file) {
    std::ofstream out(output_file);
    if (!out.is_open()) {
        throw std::runtime_error("Could not open output file: " + output_file);
    }

    out << "cluster_a,cluster_b,distance,size" << std::endl;
    for (const auto& node : dendrogram) {
        out << node.cluster_a << "," << node.cluster_b << ","
            << node.distance << "," << node.size << std::endl;
    }
    out.close();
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string input_file = argv[1];
    std::string output_file = "";
    std::string format = "json";
    std::string algorithm = "champaign";
    int num_threads = std::thread::hardware_concurrency();
    bool verbose = false;

    // Parse optional arguments
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-a" && i + 1 < argc) {
            algorithm = argv[++i];
            if (algorithm != "champaign" && algorithm != "paris") {
                std::cerr << "Error: algorithm must be 'champaign' or 'paris'" << std::endl;
                return 1;
            }
        } else if (arg == "-o" && i + 1 < argc) {
            output_file = argv[++i];
        } else if (arg == "-f" && i + 1 < argc) {
            format = argv[++i];
            if (format != "json" && format != "csv") {
                std::cerr << "Error: format must be 'json' or 'csv'" << std::endl;
                return 1;
            }
        } else if (arg == "-t" && i + 1 < argc) {
            num_threads = std::stoi(argv[++i]);
        } else if (arg == "-v") {
            verbose = true;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    std::cout << "Running " << algorithm << " algorithm on: " << input_file << std::endl;

    try {
        // Load graph
        auto start = std::chrono::high_resolution_clock::now();
        Graph graph = load_undirected_tsv_edgelist_parallel(input_file, num_threads, verbose);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "Graph loaded in " << duration.count() << " ms" << std::endl;
        std::cout << "Nodes: " << graph.num_nodes << ", Edges: " << graph.num_edges << std::endl;

        // Run selected algorithm
        start = std::chrono::high_resolution_clock::now();
        std::vector<DendrogramNode> dendrogram;
        if (algorithm == "champaign") {
            dendrogram = champaign(graph, verbose);
        } else {
            dendrogram = paris(graph, verbose);
        }
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << algorithm << " completed in " << duration.count() << " ms" << std::endl;
        std::cout << "Dendrogram size: " << dendrogram.size() << std::endl;

        // Save dendrogram if output file specified
        if (!output_file.empty()) {
            if (format == "json") {
                // Capitalize algorithm name for JSON
                std::string algo_name = algorithm;
                algo_name[0] = std::toupper(algo_name[0]);
                save_dendrogram_json(dendrogram, graph.num_nodes, graph.num_edges, output_file, algo_name);
                std::cout << "Dendrogram saved to: " << output_file << " (JSON format)" << std::endl;
            } else {
                save_dendrogram_csv(dendrogram, output_file);
                std::cout << "Dendrogram saved to: " << output_file << " (CSV format)" << std::endl;
            }
        }

        // Print first few dendrogram entries as sample
        std::cout << "\nFirst 5 dendrogram entries:" << std::endl;
        std::cout << "cluster_a, cluster_b, distance, size" << std::endl;
        for (size_t i = 0; i < std::min(size_t(5), dendrogram.size()); ++i) {
            std::cout << dendrogram[i].cluster_a << ", "
                      << dendrogram[i].cluster_b << ", "
                      << dendrogram[i].distance << ", "
                      << dendrogram[i].size << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
