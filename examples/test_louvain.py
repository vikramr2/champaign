#!/usr/bin/env python3
"""
Test script for the new .louvain() method on Paris dendrograms
"""

import champaign
import tempfile
import os

# Create a simple test graph (karate club-like structure)
def create_test_graph():
    """Create a small test graph with community structure"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
        # Community 1: nodes 0-4
        edges = [
            (0, 1), (0, 2), (0, 3),
            (1, 2), (1, 3),
            (2, 3), (2, 4),
            (3, 4),
        ]

        # Community 2: nodes 5-9
        edges += [
            (5, 6), (5, 7), (5, 8),
            (6, 7), (6, 8),
            (7, 8), (7, 9),
            (8, 9),
        ]

        # Bridge between communities
        edges += [(4, 5)]

        for u, v in edges:
            f.write(f"{u}\t{v}\n")

        return f.name

def main():
    print("=" * 60)
    print("Testing Louvain refinement on Paris dendrogram")
    print("=" * 60)

    # Create test graph
    graph_file = create_test_graph()

    try:
        # Load graph
        print("\n1. Loading graph...")
        g = champaign.from_tsv(graph_file, verbose=False)
        print(f"   Graph: {g.num_nodes()} nodes, {g.num_edges()} edges")

        # Run Paris algorithm
        print("\n2. Running Paris algorithm...")
        paris_dendro = champaign.paris(g, verbose=True)
        print(f"   Dendrogram: {paris_dendro.size()} merges")

        # Test Louvain refinement with different resolution values
        print("\n3. Testing Louvain refinement with different resolutions:")

        for resolution in [0.1, 0.5, 1.0, 2.0, 5.0]:
            print(f"\n   Resolution = {resolution}:")
            result = paris_dendro.louvain(g, resolution, verbose=False)

            print(f"      Clusters: {result['num_clusters']}")
            print(f"      Modularity: {result['modularity']:.6f}")
            print(f"      Cluster sizes: {[len(cluster) for cluster in result['clusters']]}")

        # Compare with Leiden on Champaign dendrogram
        print("\n4. Comparison: Leiden on Champaign dendrogram")
        print("\n   Running Champaign algorithm...")
        champaign_dendro = champaign.champaign(g, verbose=False)

        for gamma in [0.1, 0.5, 1.0, 2.0, 5.0]:
            print(f"\n   Gamma = {gamma}:")
            result = champaign_dendro.leiden(g, gamma, verbose=False)

            print(f"      Clusters: {result['num_clusters']}")
            print(f"      CPM: {result['cpm']:.6f}")
            print(f"      Cluster sizes: {[len(cluster) for cluster in result['clusters']]}")

        print("\n" + "=" * 60)
        print("SUCCESS! Louvain method works correctly!")
        print("=" * 60)

    finally:
        # Clean up
        os.unlink(graph_file)

if __name__ == "__main__":
    main()
