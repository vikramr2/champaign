#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include "../lib/data_structures/graph.h"
#include "../lib/io/graph_io.h"
#include "../lib/algorithm/paris.h"

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <command> [options]" << std::endl;
    std::cout << "\nCommands:" << std::endl;
    std::cout << "  paris <input_file> [output_file]  - Run PARIS algorithm on graph" << std::endl;
    std::cout << "\nOptions:" << std::endl;
    std::cout << "  -t <num_threads>  - Number of threads (default: hardware concurrency)" << std::endl;
    std::cout << "  -v                - Verbose output" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string command = argv[1];

    if (command == "paris") {
        if (argc < 3) {
            std::cerr << "Error: paris command requires input file" << std::endl;
            print_usage(argv[0]);
            return 1;
        }

        std::string input_file = argv[2];
        std::string output_file = "";
        int num_threads = std::thread::hardware_concurrency();
        bool verbose = false;

        // Parse optional arguments
        for (int i = 3; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "-t" && i + 1 < argc) {
                num_threads = std::stoi(argv[++i]);
            } else if (arg == "-v") {
                verbose = true;
            } else if (output_file.empty()) {
                output_file = arg;
            }
        }

        std::cout << "Running PARIS algorithm on: " << input_file << std::endl;

        // Load graph
        auto start = std::chrono::high_resolution_clock::now();
        Graph graph = load_undirected_tsv_edgelist_parallel(input_file, num_threads, verbose);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "Graph loaded in " << duration.count() << " ms" << std::endl;
        std::cout << "Nodes: " << graph.num_nodes << ", Edges: " << graph.num_edges << std::endl;

        // Run PARIS algorithm
        start = std::chrono::high_resolution_clock::now();
        std::vector<DendrogramNode> dendrogram = paris(graph, verbose);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "PARIS completed in " << duration.count() << " ms" << std::endl;
        std::cout << "Dendrogram size: " << dendrogram.size() << std::endl;

        // Save dendrogram if output file specified
        if (!output_file.empty()) {
            std::ofstream out(output_file);
            if (out.is_open()) {
                out << "cluster_a,cluster_b,distance,size" << std::endl;
                for (const auto& node : dendrogram) {
                    out << node.cluster_a << "," << node.cluster_b << ","
                        << node.distance << "," << node.size << std::endl;
                }
                out.close();
                std::cout << "Dendrogram saved to: " << output_file << std::endl;
            } else {
                std::cerr << "Error: Could not open output file: " << output_file << std::endl;
                return 1;
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

    } else {
        std::cerr << "Unknown command: " << command << std::endl;
        print_usage(argv[0]);
        return 1;
    }

    return 0;
}
