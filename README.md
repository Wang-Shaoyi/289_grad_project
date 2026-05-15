# Beam Clustering via GNN + GMM

Unsupervised structural beam clustering using a contrastive graph neural network (GAT) followed by Gaussian Mixture Model clustering.

## Pipeline

```
Input data → classify beam types → train GNN encoder → GMM clustering → outputs
```

## Input Data

Place the following files in a subfolder under the project root (e.g., `03_building_original/01_welton/`):

| File | Description |
|------|-------------|
| `graph_line.json` | Primary input: nodes, supports, beam topology, mechanical properties |
| `nodes.csv` | Fallback: node coordinates and support flags |
| `line_nodes_beams.csv` | Fallback: beam definitions and mechanical properties |
| `line_edges.csv` | Fallback: beam adjacency (line graph edges) |

## Usage

**Step 1 — Configure** (first cell)

```python
DATA_FOLDER = r"03_building_original/01_welton"  # path to your data
K_LIST      = [120]                               # number of clusters
CATEGORIZE  = 1   # 0=all together, 1=split column/girder/beam, 2=+bracing
EPOCHS_V1   = 1000
DEVICE      = "cuda"                              # or "cpu"
```

Key mechanical feature flags (`USE_COMPRESSION`, `USE_TENSION`, `USE_SHEAR_Z`, etc.) toggle which load cases are included as features.

**Step 2 — Run** (Run section)

```python
Zs, data = unsup_mp_gmm_pipeline()
```

This trains the GNN and runs GMM clustering. Progress prints every 50 epochs.

**Step 3 — Visualize** (Visualization section)

Set `CLUSTER_FILE` to the output CSV, then run the cells to get:
- Interactive 3D plot of beams colored by cluster (Plotly)
- 2D PCA scatter plot of embeddings

**Step 4 — Analyze** (Analysis section)

Box plots of mechanical properties (compression, moment, shear, etc.) per cluster.

## Outputs

| File | Description |
|------|-------------|
| `beam_embeddings_mp.csv` | Learned GNN embeddings (one row per merged beam) |
| `beam_clusters_mp_gmm_k{K}.csv` | Cluster assignments + soft probabilities (one row per source beam) |

## Requirements

```
torch, torch_geometric, scikit-learn, pandas, numpy, plotly, matplotlib
```

GPU recommended for large models (set `DEVICE = "cuda"`).
