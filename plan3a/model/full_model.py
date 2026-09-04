"""
Full Model: Plan 3a — Hypergraph Concept Bottleneck GNN.

End-to-end composition of all modules:
  Stage 1: SheafHGNN (patch hypergraph → node embeddings)
  Stage 2: ConceptBottleneck (embeddings → interpretable concepts)
  Stage 2.5 (optional): MultiGranularTree (hierarchical coarsening)
  Stage 3: MultiModalFusion (concepts + clinical → fused embedding)
  Stage 4: SurvivalHead (fused → hazard prediction)

Training loss:
  L = (1/2σ₁²)·L_survival + (1/2σ₂²)·L_concept + log(σ₁σ₂)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from plan3a.config import SHEAF_HGNN_DIM, SHEAF_HGNN_LAYERS, NUM_CONCEPTS
from plan3a.model.sheaf_hgnn import SheafHGNN, GraphPooling
from plan3a.model.concept_bottleneck import ConceptBottleneck
from plan3a.model.fusion import MultiModalFusion
from plan3a.model.tree import MultiGranularTree


class SurvivalHead(nn.Module):
    """
    Survival prediction head.

    Predicts discrete hazard probabilities for time-binned survival.
    Uses NLL survival loss (MRePath Eq. 1).

    Output: (B, num_bins) hazard logits per time bin.
    """

    def __init__(self, in_dim: int, num_bins: int = 4, dropout: float = 0.2):
        super().__init__()
        self.num_bins = num_bins
        self.head = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_bins),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_dim) → (B, num_bins) hazard logits"""
        return self.head(x)


class NLLSurvivalLoss(nn.Module):
    """
    Negative log-likelihood survival loss for discrete time bins.

    For each sample, given hazard probabilities h_k for bin k:
      - If event observed in bin k: -log(h_k) - Σ_{j<k} log(1 - h_j)
      - If censored in bin k: -Σ_{j≤k} log(1 - h_j)

    This handles right-censored survival data properly.
    """

    def __init__(self, num_bins: int = 4):
        super().__init__()
        self.num_bins = num_bins

    def forward(
        self,
        hazard_logits: torch.Tensor,
        survival_time: torch.Tensor,
        event: torch.Tensor,
        time_bins: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hazard_logits: (B, num_bins) raw hazard logits
            survival_time: (B,) time in days
            event: (B,) 1=deceased, 0=censored
            time_bins: (num_bins,) bin boundaries in days

        Returns:
            loss: scalar NLL survival loss
        """
        hazard = torch.sigmoid(hazard_logits)  # (B, K)
        hazard = hazard.clamp(1e-7, 1 - 1e-7)

        B = hazard.shape[0]

        # Assign each sample to a time bin
        bin_idx = torch.bucketize(survival_time, time_bins) - 1
        bin_idx = bin_idx.clamp(0, self.num_bins - 1)

        loss = torch.tensor(0.0, device=hazard.device)

        for i in range(B):
            k = bin_idx[i].item()
            is_event = event[i].item()

            # Sum log(1 - h_j) for all bins before the event/censoring bin
            survival_part = torch.tensor(0.0, device=hazard.device)
            for j in range(k):
                survival_part += torch.log(1 - hazard[i, j])

            if is_event == 1:
                # Event in bin k: -log(h_k) - survival_part
                loss += -torch.log(hazard[i, k]) - survival_part
            else:
                # Censored: -survival_part - log(1 - h_k)
                loss += -survival_part - torch.log(1 - hazard[i, k])

        return loss / max(B, 1)


class Plan3aModel(nn.Module):
    """
    Full Plan 3a Model: Hypergraph Concept Bottleneck GNN.

    End-to-end architecture:
      MRI Patches → SheafHGNN → ConceptBottleneck → MultiModalFusion → SurvivalHead

    All predictions are traceable through interpretable concepts.
    """

    def __init__(
        self,
        patch_dim: int = 1536,
        embed_dim: int = SHEAF_HGNN_DIM,
        num_layers: int = SHEAF_HGNN_LAYERS,
        num_concepts: int = NUM_CONCEPTS,
        clinical_dim: int = None,
        num_survival_bins: int = 4,
        use_hecrl: bool = True,
        residual_bypass: bool = False,
        use_fusion: bool = True,
        use_tree: bool = False,
        tree_levels: int = 3,
    ):
        super().__init__()
        self.use_fusion = use_fusion
        self.use_tree = use_tree

        # Stage 1: Sheaf Hypergraph Neural Network
        self.shgnn = SheafHGNN(
            in_dim=patch_dim,
            embed_dim=embed_dim,
            num_layers=num_layers,
        )

        # Graph pooling: node embeddings → graph embedding
        self.pooler = GraphPooling(embed_dim)

        # Stage 2: Concept Bottleneck
        self.concept_bottleneck = ConceptBottleneck(
            embed_dim=embed_dim,
            num_concepts=num_concepts,
            use_hecrl=use_hecrl,
            residual_bypass=residual_bypass,
        )

        # Concept pooling: pool concept activations to graph level
        bottleneck_dim = self.concept_bottleneck.output_dim
        self.concept_pool = nn.Sequential(
            nn.Linear(bottleneck_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        # Stage 2.5: Multi-Granular Tree (optional — E5 experiment)
        if use_tree:
            self.tree = MultiGranularTree(
                dim=embed_dim, num_levels=tree_levels, coarsen_ratio=0.25,
            )
        else:
            self.tree = None

        # Stage 3: Multimodal Fusion (optional)
        if use_fusion:
            self.fusion = MultiModalFusion(embed_dim, clinical_dim)
            survival_in_dim = embed_dim
        else:
            self.fusion = None
            survival_in_dim = embed_dim

        # Stage 4: Survival Head
        self.survival_head = SurvivalHead(survival_in_dim, num_survival_bins)

        # Learned task weights (Kendall et al. 2018)
        # log(σ²) for each task — initialized to equal weighting
        self.log_var_survival = nn.Parameter(torch.tensor(0.0))
        self.log_var_concept = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        node_features: torch.Tensor,
        hyperedge_index: torch.Tensor,
        num_nodes: int,
        num_edges: int,
        concept_targets: Optional[torch.Tensor] = None,
        clinical_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.

        Args:
            node_features: (N, 1536) flattened patch features
            hyperedge_index: (2, E) hypergraph incidence
            num_nodes: N
            num_edges: number of hyperedges
            concept_targets: (N, 8) precomputed concept GT
            clinical_features: (1, clinical_dim) clinical feature vector

        Returns:
            dict with all intermediate outputs for interpretability
        """
        # ── Stage 1: SheafHGNN ───────────────────────────────────────
        node_embeds = self.shgnn(
            node_features, hyperedge_index, num_nodes, num_edges
        )  # (N, embed_dim)

        # ── Stage 2: Concept Bottleneck ──────────────────────────────
        concept_out = self.concept_bottleneck(node_embeds, concept_targets)
        # concept_out["bottleneck_output"]: (N, concept_dim)
        # concept_out["concepts"]: (N, 8)
        # concept_out["concept_loss"]: scalar

        # Pool concepts to graph level
        concept_graph = self.concept_pool(concept_out["bottleneck_output"])  # (N, D)

        # ── Stage 2.5: Multi-Granular Tree (optional) ────────────────
        tree_output = None
        if self.use_tree and self.tree is not None:
            tree_output = self.tree(
                concept_graph, hyperedge_index, num_nodes, num_edges
            )
            graph_embed = tree_output["fused_embed"]  # (1, embed_dim)
        else:
            # Standard attention-weighted pooling over nodes
            graph_embed = self.pooler(concept_graph)  # (1, embed_dim)

        # ── Stage 3: Multimodal Fusion ───────────────────────────────
        if self.use_fusion and clinical_features is not None:
            if clinical_features.dim() == 1:
                clinical_features = clinical_features.unsqueeze(0)
            fusion_out = self.fusion(graph_embed, clinical_features)
            fused = fusion_out["fused"]  # (1, embed_dim)
            w_img = fusion_out["w_img"]
            w_clin = fusion_out["w_clin"]
        else:
            fused = graph_embed
            w_img = torch.tensor([1.0], device=node_features.device)
            w_clin = torch.tensor([0.0], device=node_features.device)

        # ── Stage 4: Survival Head ───────────────────────────────────
        hazard_logits = self.survival_head(fused)  # (1, num_bins)

        result = {
            # Core outputs
            "hazard_logits": hazard_logits,
            "graph_embed": graph_embed,
            "fused_embed": fused,
            # Concept outputs
            "concepts": concept_out["concepts"],
            "concept_raw": concept_out["concept_raw"],
            "concept_loss": concept_out["concept_loss"],
            # Node-level
            "node_embeddings": node_embeds,
            # Fusion weights
            "w_img": w_img,
            "w_clin": w_clin,
        }

        # Add tree outputs if available
        if tree_output is not None:
            result["routing_weights"] = tree_output["routing_weights"]
            result["tree_level_data"] = tree_output["level_data"]
            result["num_tree_levels"] = tree_output["num_levels"]

        return result

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        survival_time: torch.Tensor,
        event: torch.Tensor,
        time_bins: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss with learned task weights.

        L = (1/2σ₁²)·L_survival + (1/2σ₂²)·L_concept + log(σ₁σ₂)
        """
        # Survival loss
        nll_loss = NLLSurvivalLoss(num_bins=outputs["hazard_logits"].shape[-1])
        l_survival = nll_loss(
            outputs["hazard_logits"], survival_time, event, time_bins
        )

        # Concept loss (from bottleneck forward pass)
        l_concept = outputs["concept_loss"]

        # Learned weighting (Kendall)
        precision_surv = torch.exp(-self.log_var_survival)
        precision_conc = torch.exp(-self.log_var_concept)

        total_loss = (
            0.5 * precision_surv * l_survival
            + 0.5 * precision_conc * l_concept
            + 0.5 * (self.log_var_survival + self.log_var_concept)
        )

        return {
            "total_loss": total_loss,
            "survival_loss": l_survival,
            "concept_loss": l_concept,
            "log_var_survival": self.log_var_survival,
            "log_var_concept": self.log_var_concept,
        }
