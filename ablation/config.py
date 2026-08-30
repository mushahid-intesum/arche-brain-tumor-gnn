"""
Ablation Study Configuration

Defines all ablation configurations, seeds, and training hyperparameters.
The AblationConfig controls which components (SV, topology, OCN) are active.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AblationConfig:
    """Controls which components are active for a given ablation run.

    Attributes:
        use_sv_aggregation: If True, use IntraNodeAggregator (Transformer)
            to aggregate supervoxel features into node embeddings.
            If False, use flat 35-dim features zero-padded to 68.
        use_ocn_features: If True, compute OCN residual and path-normalized CN
            in structural features (dims 3 and 4). If False, set those dims to 0.
        use_intra_topology: If True, compute and concatenate the 4-dim
            intra-node topology features (CN density, connectivity, degree var,
            spectral gap). If False, zero-pad those 4 dims.
        edge_strategy: "compatibility_only" or "knn_filtered". Controls how
            inter-node edges are constructed (see gnn.build_inter_edges).
        num_gnn_layers: Number of GATv2 encoder layers (2 or 3).
        name: Human-readable identifier for this configuration.
    """
    use_sv_aggregation: bool = True
    use_ocn_features: bool = True
    use_intra_topology: bool = True
    edge_strategy: str = "compatibility_only"
    num_gnn_layers: int = 2
    name: str = "full"


# ── Preset Configurations ─────────────────────────────────────────────

CONFIGS = {
    # A: Raw GNN baseline. No SV aggregation, no topology, no OCN.
    #    Uses flat 35-dim features, zero-padded to 68.
    "A_baseline": AblationConfig(
        use_sv_aggregation=False,
        use_ocn_features=False,
        use_intra_topology=False,
        edge_strategy="compatibility_only",
        num_gnn_layers=2,
        name="A_baseline",
    ),

    # B: GNN + SV aggregation + topology features. No OCN.
    #    Uses 64-dim Transformer embedding + 4-dim topology = 68-dim.
    "B_sv_topo": AblationConfig(
        use_sv_aggregation=True,
        use_ocn_features=False,
        use_intra_topology=True,
        edge_strategy="compatibility_only",
        num_gnn_layers=2,
        name="B_sv_topo",
    ),

    # B': GNN + SV aggregation only. No topology, no OCN.
    #     Uses 64-dim Transformer embedding, zero-padded to 68.
    "B_prime_sv": AblationConfig(
        use_sv_aggregation=True,
        use_ocn_features=False,
        use_intra_topology=False,
        edge_strategy="compatibility_only",
        num_gnn_layers=2,
        name="B_prime_sv",
    ),

    # C: GNN + OCN only. No SV aggregation, no topology.
    #    Uses flat 35-dim features, zero-padded to 68. OCN structural feats active.
    "C_ocn_only": AblationConfig(
        use_sv_aggregation=False,
        use_ocn_features=True,
        use_intra_topology=False,
        edge_strategy="compatibility_only",
        num_gnn_layers=2,
        name="C_ocn_only",
    ),

    # D: Full model. SV aggregation + topology + OCN.
    "D_full": AblationConfig(
        use_sv_aggregation=True,
        use_ocn_features=True,
        use_intra_topology=True,
        edge_strategy="compatibility_only",
        num_gnn_layers=2,
        name="D_full",
    ),
}

# Ordering for reports (alphabetical by key)
CONFIG_ORDER = ["A_baseline", "B_sv_topo", "B_prime_sv", "C_ocn_only", "D_full"]

# ── Training Hyperparameters ──────────────────────────────────────────

SEEDS = [42, 123, 256, 512, 1024]
EPOCHS = 80
LR = 5e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

# ── Output ────────────────────────────────────────────────────────────

OUTPUT_DIR = Path("brats_outputs/ablation")
