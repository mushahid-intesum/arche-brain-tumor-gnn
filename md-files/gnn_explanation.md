# GNN Edge Prediction Pipeline -- Complete Explanation

## 1. From 3D MRI Volumes to Graphs

The GNN pipeline receives 3D MRI volumes (4 modalities: T1-native, T1-contrast, T2-weighted, T2-FLAIR) and predicted segmentation masks from Phase 3 (DeepLabV3+ segmentation). Each segmentation mask labels voxels as one of three tissue types: NCR (Necrotic Core, label 1), ED (Edema, label 2), or ET (Enhancing Tumor, label 3). The goal is to construct a graph where each node represents a 3D tissue region, internal supervoxels capture sub-region heterogeneity, and edges represent structural connectivity between tissue regions.

### 1.1 Supervoxel Generation

Before building the graph, the raw 4-channel MRI volume is partitioned into supervoxels using 3D SLIC (Simple Linear Iterative Clustering). SLIC groups spatially adjacent voxels with similar intensity profiles into compact 3D clusters. Each cluster is called a supervoxel.

The algorithm starts by placing seed points on a regular 3D grid, then iteratively assigns each voxel to the nearest seed in a combined spatial-intensity distance metric. The result is approximately 500 supervoxels, each containing roughly 20 to 100 voxels.

Background supervoxels (those that fall mostly outside the tumor mask) are pruned. Each remaining supervoxel is assigned to the segmentation component (NCR, ED, or ET connected component) it overlaps with most.

### 1.2 Supervoxel Features (25 dimensions per supervoxel)

Each supervoxel receives a 25-dimensional feature vector:

**Intensity Statistics (dims 0 to 15)**: For each of the 4 MRI modalities, 4 statistics are computed from the voxel values inside the supervoxel. This gives 4 modalities times 4 statistics equals 16 dimensions.

The four statistics per modality are:
- Mean: the average voxel intensity, capturing the characteristic brightness on that modality.
- Standard deviation: how spread out the values are, capturing internal heterogeneity.
- Range: max minus min, capturing the dynamic range of the signal.
- Skewness: asymmetry of the intensity distribution, detecting partial volume effects.

**Volume and Shape (dims 16 to 21)**: Normalized volume (voxel count divided by total tumor volume), surface area estimate, compactness ratio (how close to a sphere), and bounding box dimensions.

**Relative Positional Encoding (dims 22 to 24)**: Three values encoding the supervoxel's position relative to the centroid of its parent segmentation component. This gives the Transformer context about where each supervoxel sits within the larger tissue region.

### 1.3 Hierarchical Graph Construction

The `build_hierarchical_graph` function in `gnn.py` builds one graph per patient case.

**Nodes**: Each 3D connected component in the segmentation mask (a contiguous blob of one tissue type) becomes a node. A typical graph has 5 to 30 nodes depending on tumor complexity.

**Intra-node structure**: Each node contains a set of supervoxels. A KNN graph (k=3) is built over the supervoxels within each node, connecting spatially adjacent supervoxels. This internal graph captures the spatial layout of sub-regions within each tissue component.

**Inter-node edges**: Nodes are connected using KNN (k=5) based on 3D centroid distances, filtered by tissue compatibility. Two nodes are connected only if their tissue types are compatible:
- Same type (NCR to NCR, ED to ED, ET to ET)
- Adjacent types (NCR to ED, ED to ET)
- NCR to ET (necrotic core is directly surrounded by the enhancing rim in glioblastoma)

These rules encode known spatial relationships in glioblastoma anatomy.

**Edge attributes (4 dimensions per edge)**:

| Dim | Feature | What it captures |
|-----|---------|-----------------|
| 0 | Normalized distance | How far apart the two region centroids are |
| 1 | Normalized angle | Directional relationship between regions |
| 2 | Z gap | Depth distance between regions (normalized by volume depth) |
| 3 | Same tissue flag | 1.0 if both endpoints are the same tissue type, 0.0 otherwise |

The resulting PyG Data object contains standard fields (node features, edge index, positions, tissue labels) plus hierarchical fields: `sv_features` (list of tensors, one per node, containing supervoxel feature matrices), `sv_edge_indices` (internal KNN graphs per node), and `n_svs_per_node` (supervoxel counts).

---

## 2. Supervoxel Aggregation (IntraNodeAggregator)

Each node has a variable number of supervoxels (denoted K_i for node i). The `IntraNodeAggregator` converts this variable-length set into a fixed 64-dimensional node embedding using a Transformer architecture with a CLS (classification) token. This design is inspired by the SVGFormer paper.

The processing steps for each node are:

1. **Linear projection**: Each 25-dim supervoxel feature vector is projected to 64 dimensions through a learned linear layer.

2. **CLS token prepending**: A learnable 64-dim CLS token is prepended to the sequence. The resulting sequence has length K_i + 1.

3. **Transformer encoding**: The sequence passes through a 2-layer Transformer encoder with 4 attention heads, GELU activation, feedforward dimension of 128, and dropout of 0.1. Each supervoxel attends to every other supervoxel in the same node.

4. **Output extraction**: The CLS token's output is passed through a final linear layer to produce the 64-dim node embedding.

5. **Attention weights**: For explainability, dot-product attention weights between each supervoxel and the CLS output are computed and softmax-normalized. These weights indicate how much each supervoxel contributed to the final node representation.

The advantage over simple mean or max pooling is that the Transformer learns to selectively attend to diagnostically important supervoxels. For example, in a large edema region, the model might focus on supervoxels at the tumor-edema boundary where infiltration occurs, rather than averaging over the entire uniform edema volume.

---

## 3. OCN Structural Features (5 dimensions per edge)

After the GATv2 encoder produces node embeddings, the `StructuralFeatureComputer` computes topology-based features for candidate edges. OCN stands for Orthogonalized Common Neighbors, extending classical common neighbor heuristics by removing redundant information already captured by the node embeddings.

For each candidate edge between nodes i and j, five features are computed:

| Dim | Feature | Formula | What it measures |
|-----|---------|---------|-----------------|
| 0 | CN count | \|N(i) intersection N(j)\| / max_degree | How many neighbors i and j share, normalized by max degree |
| 1 | Jaccard | \|N(i) intersection N(j)\| / \|N(i) union N(j)\| | Overlap relative to total neighborhood size |
| 2 | Adamic-Adar | Sum of 1/log(deg(w)) for w in CN(i,j) | Weighted CN count, rare shared neighbors count more |
| 3 | OCN residual | norm(cn_signal - projection onto span(z_i, z_j)) | Structural information NOT already captured by endpoint embeddings |
| 4 | Path-normalized CN | \|CN(i,j)\| / (2-hop reachability count) | CN count corrected for local graph density |

**OCN Residual (dim 3)** is the key contribution from the OCN paper. The computation works as follows:

1. Compute the mean embedding of all common neighbors. This is the CN signal.
2. Project this CN signal onto the 2-dimensional subspace spanned by the embeddings of nodes i and j (using QR decomposition).
3. Subtract the projection from the CN signal. The remainder is the residual.
4. Take the Euclidean norm of this residual.

If the residual is zero, the common neighbors encode nothing beyond what the endpoint embeddings already capture. A large residual means the common neighbors carry structural information that the endpoints alone cannot explain, making the edge structurally significant.

**Path-Normalized CN (dim 4)** corrects the raw CN count by dividing by the number of 2-hop paths between i and j. In dense subgraphs, raw CN counts are inflated because many 2-hop paths exist between any pair of nodes. Path normalization asks: "Is the observed CN count surprisingly high, or just expected given local density?"

Additionally, the actual node indices of common neighbors are stored so their embeddings can be pooled in the decoder.

---

## 4. Intra-Node Topology (4 dimensions per node)

The `compute_intra_node_topology` method computes a 4-dimensional topological fingerprint for each node based on its internal supervoxel graph:

| Dim | Feature | What it captures |
|-----|---------|-----------------|
| 0 | CN density | Average common neighbor fraction across all supervoxel pairs inside the node |
| 1 | Connectivity ratio | Fraction of possible supervoxel pairs that share at least one common neighbor |
| 2 | Degree variance | Normalized variance of supervoxel degrees, measuring structural regularity |
| 3 | Spectral gap | Difference between the two largest eigenvalues of the normalized Laplacian |

The spectral gap is particularly informative. A large spectral gap means the supervoxel graph has a strong community structure (well-separated clusters of supervoxels). A small spectral gap means the graph is more uniformly connected. In brain tumors, enhancing tumor regions tend to have higher spectral gaps because they contain distinct sub-populations of voxels (the bright enhancing rim versus the inner transition zone).

These 4 topology features are concatenated with the 64-dim aggregator output to form the final 68-dim node feature vector.

---

## 5. Model Architecture: EdgePredictor

The model follows the "encode then decode" architecture. The encoder transforms node features into embeddings. The decoder uses those embeddings plus structural features to predict edge probabilities.

### 5.1 GATv2Encoder -- 3-Layer GATv2 with Residual Connections

The `GATv2Encoder` processes the 68-dim node features through three layers of Graph Attention Network version 2 (GATv2):

1. **Input projection**: Linear layer from 68 to 128 dimensions.

2. **Three GATv2 layers**: Each layer uses 4 attention heads (32 dims per head, concatenated to 128). The 4-dim edge attributes (distance, angle, z gap, same tissue) are incorporated as attention biases. Each layer includes LayerNorm and a residual connection. Between layers, ELU activation and dropout (0.2) are applied.

3. **Output projection**: Linear layer from 128 to 64 dimensions.

GATv2 is used instead of the original GAT because GATv2 computes attention as a_transpose times LeakyReLU(W times [h_i concatenated with h_j]). The attention score depends on both source and target features. In the original GAT, the ranking of attention scores for a fixed target is the same regardless of the source, which is a known expressiveness limitation that GATv2 fixes.

LayerNorm is used instead of BatchNorm because our graphs are small (5 to 30 nodes) and variable-sized. BatchNorm computes statistics across all nodes in a batch, which is unstable with varying graph sizes. LayerNorm normalizes each node independently.

Three layers give each node a 3-hop receptive field, which is typically enough to cover the entire tumor graph.

### 5.2 MultiSignalDecoder -- 6-Signal Edge Decoder

For each candidate edge between nodes i and j, the `MultiSignalDecoder` constructs a 416-dimensional edge representation from six independent signals:

**Signal 1: Hadamard product (64 dims)**
Element-wise multiplication z_i times z_j. Each dimension is large when both nodes have large values, capturing per-dimension feature agreement.

**Signal 2: Concatenation (128 dims)**
[z_i, z_j] concatenated. Preserves full information from both nodes, allowing the MLP to learn asymmetric patterns (for example, "a large region connected to a small region" is different from "two large regions connected").

**Signal 3: Common Neighbor Pool (64 dims)**
Mean embedding of all common neighbor nodes. If nodes i and j share neighbors w1 and w2, this signal is the average of z_w1 and z_w2. This encodes the topological context of the edge. If the common neighbor set is empty, this is a zero vector.

**Signal 4: Structural Embedding (64 dims)**
The 5-dim OCN structural features (CN count, Jaccard, Adamic-Adar, OCN residual, path-normalized CN) projected to 64 dims through a learned linear layer.

**Signal 5: Tissue Pair Embedding (64 dims)**
A learned embedding for the ordered tissue type pair. There are 9 possible pairs (3 source types times 3 target types). Each pair has its own 64-dim learned vector, allowing the model to learn that certain tissue pairs (like ET to NCR) are intrinsically more likely to be connected than others.

**Signal 6: Edge Type Embedding (32 dims)**
A learned embedding for whether the edge is intra-slice (both endpoints in the same axial depth) or inter-slice (endpoints at different depths). This allows different link prediction strategies for lateral versus depth connections.

All six signals are concatenated to form 416 dimensions (64 + 128 + 64 + 64 + 64 + 32). This passes through a three-layer MLP:
- LayerNorm on the 416-dim input
- Linear(416 to 128) + GELU + Dropout(0.3)
- Linear(128 to 64) + GELU + Dropout(0.2)
- Linear(64 to 1) producing a single logit

The logit is passed through sigmoid to obtain the link probability.

### 5.3 Why This Architecture

| Approach | Expressiveness | Cost | Status |
|----------|---------------|------|--------|
| Dot product decoder (GAE/VGAE) | Low, only captures linear similarity | O(1) per edge | Not used, too simple |
| Concat MLP (no structural context) | Medium, no CN information | O(1) per edge | Not used, misses topology |
| SEAL subgraph extraction | Very high, full local structure | O(k-hop subgraph) per edge | Not used, too expensive |
| OCN multi-signal (current) | High, CN pooling + OCN residuals + tissue context | O(CN lookup) per edge | Current architecture |

---

## 6. Training

### 6.1 Degree-Biased Negative Sampling

Instead of sampling negative edges uniformly at random, node pairs are sampled proportional to their degree:

```
probability(node n as negative endpoint) proportional to degree(n) + 1
```

Uniform sampling creates easy negatives (two random nodes that are obviously far apart and not connected). Degree-biased sampling creates harder negatives involving hub nodes, which are structurally more difficult to distinguish from true positives.

### 6.2 Loss Function

Standard binary cross-entropy with logits on positive and negative edges:

```
loss = BCE(positive_predictions, 1) + BCE(negative_predictions, 0)
```

### 6.3 Optimization

- **AdamW**: Learning rate 5e-4, weight decay 1e-4.
- **OneCycleLR scheduler**: Ramps learning rate up then decays, preventing early overfitting.
- **Gradient clipping**: Maximum norm of 1.0, stabilizing training on small variable-size graphs.
- **Per-graph processing**: Each graph is processed individually (no batching) due to variable graph sizes and VRAM constraints (RTX 3060 12GB).

### 6.4 Evaluation Metrics

- **AUC-ROC**: How well the model separates positive from negative edges across all thresholds.
- **Average Precision**: Area under the precision-recall curve, more informative when the positive/negative ratio is imbalanced.
- **Intra-slice AUC**: Performance on edges where both endpoints are at the same depth.
- **Inter-slice AUC**: Performance on edges crossing depth boundaries, which is generally harder.
- **Tissue-pair scores**: Mean prediction confidence per tissue type pair, revealing which tissue relationships the model is most and least confident about.

---

## 7. 3-Level Explanation System

The `HierarchicalExplainer` class generates multi-granularity explanations for each predicted edge.

**Level 1: Structural Evidence**
Reports the OCN structural features for the predicted edge: common neighbor count, Jaccard coefficient, Adamic-Adar index, OCN residual, and path-normalized CN. Also reports the tissue types of both endpoints, their spatial distance, and whether the edge is intra-slice or inter-slice.

**Level 2: Supervoxel Attribution**
Reports the top-3 supervoxels in each endpoint node that received the highest attention weights from the IntraNodeAggregator. Also reports the attention entropy: low entropy means focused attention on specific supervoxels, high entropy means distributed attention across many supervoxels.

Example output:
```
Level 2 -- Supervoxel Attribution:
  Source (ET): focused attention (H=0.72)
    SV#3: weight=0.412
    SV#7: weight=0.289
    SV#1: weight=0.156
  Target (NCR): distributed attention (H=1.84)
    SV#12: weight=0.198
    SV#5: weight=0.171
```

**Level 3: Spatial Heatmap**
Maps the supervoxel attention weights back to voxel coordinates in the original MRI volume. For each important supervoxel, all voxels belonging to that supervoxel are painted with its attention weight value. The result is a 3D heatmap where bright regions correspond to the most important voxels for the predicted connection.

**Visualization**: The `plot_hierarchical_explanation` function generates a 4-panel figure:
- Panel 1: Graph topology with nodes colored by tissue type and the explained edge highlighted.
- Panel 2: Supervoxel attention heatmap overlaid on segmentation contours.
- Panel 3: Bar chart of the five OCN structural features.
- Panel 4: Full text explanation showing all three levels.

---

## 8. Dimension Summary

### Node Features

| Total Dims | Breakdown |
|------------|-----------|
| 68 | 64 from Transformer aggregator + 4 from intra-node topology |

The 64-dim embedding comes from the IntraNodeAggregator processing variable-length supervoxel feature sets (each 25-dim). The 4 topology dims come from the internal supervoxel graph structure (CN density, connectivity ratio, degree variance, spectral gap).

### Edge Features

| Type | Dims | Components |
|------|------|------------|
| Edge attributes | 4 | distance, angle, z gap, same tissue flag |
| Structural (OCN) | 5 | CN count, Jaccard, Adamic-Adar, OCN residual, path-normalized CN |

### Model Dimensions

| Component | Input to Output |
|-----------|-----------------|
| IntraNodeAggregator | (K_i, 25) to 64 |
| GATv2Encoder | 68 to 128 to 64 |
| MultiSignalDecoder | 416 to 128 to 64 to 1 |
| Total parameters | Approximately 500K (fits on RTX 3060 12GB) |
