from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExplainabilityConfig:
    """Controls graph construction and model architecture for one experiment."""

    edge_strategy: str          # "compatibility_only" or "knn_filtered"
    k_neighbors: int            # 0 for compatibility_only, 2-5 for knn_filtered
    use_sv_aggregation: bool
    use_ocn_features: bool
    use_intra_topology: bool
    num_gnn_layers: int         # 2 or 3
    name: str


# ── Group 1: KNN Sensitivity (full model, tested at 2 and 3 layers) ──

KNN_CONFIGS = {}

for layers in [2, 3]:
    suffix = f"_L{layers}"

    KNN_CONFIGS[f"K0_compat{suffix}"] = ExplainabilityConfig(
        edge_strategy="compatibility_only",
        k_neighbors=0,
        use_sv_aggregation=True,
        use_ocn_features=True,
        use_intra_topology=True,
        num_gnn_layers=layers,
        name=f"K0_compat{suffix}",
    )

    for k in [2, 3, 4, 5]:
        KNN_CONFIGS[f"K{k}{suffix}"] = ExplainabilityConfig(
            edge_strategy="knn_filtered",
            k_neighbors=k,
            use_sv_aggregation=True,
            use_ocn_features=True,
            use_intra_topology=True,
            num_gnn_layers=layers,
            name=f"K{k}{suffix}",
        )


# ── Group 2: Component Ablation (compatibility_only, 2 layers) ───────

ABLATION_CONFIGS = {
    "A_baseline": ExplainabilityConfig(
        edge_strategy="compatibility_only",
        k_neighbors=0,
        use_sv_aggregation=False,
        use_ocn_features=False,
        use_intra_topology=False,
        num_gnn_layers=2,
        name="A_baseline",
    ),
    "B_sv_topo": ExplainabilityConfig(
        edge_strategy="compatibility_only",
        k_neighbors=0,
        use_sv_aggregation=True,
        use_ocn_features=False,
        use_intra_topology=True,
        num_gnn_layers=2,
        name="B_sv_topo",
    ),
    "B_prime_sv": ExplainabilityConfig(
        edge_strategy="compatibility_only",
        k_neighbors=0,
        use_sv_aggregation=True,
        use_ocn_features=False,
        use_intra_topology=False,
        num_gnn_layers=2,
        name="B_prime_sv",
    ),
    "C_ocn_only": ExplainabilityConfig(
        edge_strategy="compatibility_only",
        k_neighbors=0,
        use_sv_aggregation=False,
        use_ocn_features=True,
        use_intra_topology=False,
        num_gnn_layers=2,
        name="C_ocn_only",
    ),
    "D_full": ExplainabilityConfig(
        edge_strategy="compatibility_only",
        k_neighbors=0,
        use_sv_aggregation=True,
        use_ocn_features=True,
        use_intra_topology=True,
        num_gnn_layers=2,
        name="D_full",
    ),
}

# Note: D_full is identical to K0_compat_L2. Only one model is trained.


# ── Combined ──────────────────────────────────────────────────────────

CONFIGS = {}
CONFIGS.update(KNN_CONFIGS)
CONFIGS.update(ABLATION_CONFIGS)

CONFIG_ORDER = (
    # Group 1: KNN sensitivity (2 layers)
    ["K0_compat_L2", "K2_L2", "K3_L2", "K4_L2", "K5_L2"]
    # Group 1: KNN sensitivity (3 layers)
    + ["K0_compat_L3", "K2_L3", "K3_L3", "K4_L3", "K5_L3"]
    # Group 2: Component ablation
    + ["A_baseline", "B_sv_topo", "B_prime_sv", "C_ocn_only", "D_full"]
)


# ── Post-hoc methods (applied to D_full model) ───────────────────────

POSTHOC_METHODS = ["intrinsic", "gnn_explainer", "grad_cam", "attention_only"]


# ── Training ──────────────────────────────────────────────────────────

SEEDS = [42, 123, 256, 512, 1024]
EPOCHS = 80
LR = 5e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

OUTPUT_DIR = Path("brats_outputs/explainability")
