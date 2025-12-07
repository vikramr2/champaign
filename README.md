# Champaign
Paris + CPM = Champaign (clearly)

Fast hierarchical graph clustering algorithms with tree slicing for rapid multi-resoluton snapshots.

## Introduction

An efficient hierarchical graph clustering suite with rapid multi-resolution slicing for exploring clustering snapshots at multiple resolutions:

- **Champaign**: CPM-optimized hierarchical clustering
- **Paris**: Modularity-based hierarchical clustering
- **Leiden**: CPM-based tree slicing for Champaign dendrograms
- **Louvain**: Modularity-based tree slicing for Paris dendrograms

## Setup

```bash
pip install git+https://github.com/vikramr2/champaign.git
```

## Quick Start

```python
import champaign

# Load graph from TSV edge list
g = champaign.from_tsv("graph.tsv")

# Method 1: Champaign
dendro = champaign.champaign(g)
result = dendro.leiden(g, gamma=0.5) # Generate Leiden slice for 0.5 res snapshot

# Method 2: Paris + Louvain (modularity optimization)
dendro = champaign.paris(g)
result = dendro.louvain(g, resolution=0.5) # Generate Louvain slice for 0.5 res snapshot
```

## Features

### Hierarchical Clustering

**Champaign**: Builds dendrograms using CPM (Constant Potts Model) with cluster size weighting. Optimized for detecting communities with strong internal connectivity.

**Paris**: Builds dendrograms using node degree-based distance metric. Optimized for modularity-based community structure.

### Slicing

**Leiden Slicing**: Extracts partition from dendrogram at distance `1/gamma`, then refines using CPM local moves. Returns clusters with CPM score.

**Louvain Slicing**: Extracts partition from dendrogram at distance `1/resolution`, then refines using modularity local moves. Returns clusters with modularity score.
