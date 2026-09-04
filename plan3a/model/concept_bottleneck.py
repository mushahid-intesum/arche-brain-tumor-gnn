"""
Concept Bottleneck Layer for Plan 3a (HyperCBM-inspired).

Forces the model to reason through 8 clinically-grounded concepts
before making predictions. This provides ante-hoc interpretability:
every prediction is traceable to specific concept activations.

Architecture:
  1. ConceptPredictor: SHGNN embeddings → 8 concept predictions per node
  2. HECRL: Concept-level hypergraph for inter-concept consistency
  3. BottleneckClassifier: Concepts-only → survival prediction

All concepts are self-supervised from raw imaging (no seg GT needed):
  c1: Enhancement ratio (T1-post/T1-pre)
  c2: FLAIR z-score (edema)
  c3: T2 abnormality
  c4: DTI mean diffusivity
  c5: DTI fractional anisotropy proxy
  c6: Intensity heterogeneity
  c7: Boundary complexity (from graph neighborhood)
  c8: Spatial location (z-coordinate)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from plan3a.config import NUM_CONCEPTS, EMBED_DIM, SHEAF_HGNN_DIM


class ConceptPredictor(nn.Module):
    """
    Predict concept values from SHGNN node embeddings.

    Each concept gets its own small prediction head to encourage
    disentangled representations. The predicted concepts are then
    supervised against the precomputed concept ground truth values
    from patch_extraction.py.

    Input: (N, embed_dim) node embeddings from SHGNN
    Output: (N, num_concepts) predicted concept activations
    """

    def __init__(
        self,
        embed_dim: int = SHEAF_HGNN_DIM,
        num_concepts: int = NUM_CONCEPTS,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.num_concepts = num_concepts

        # Shared trunk: reduces dimensionality before concept-specific heads
        self.shared = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Per-concept prediction heads (disentangled)
        self.concept_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
            )
            for _ in range(num_concepts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, embed_dim) → (N, num_concepts)
        """
        shared_feat = self.shared(x)  # (N, hidden_dim)
        concept_preds = []
        for head in self.concept_heads:
            concept_preds.append(head(shared_feat))  # (N, 1)
        return torch.cat(concept_preds, dim=-1)  # (N, num_concepts)


class HECRL(nn.Module):
    """
    Hypergraph-Enhanced Concept Representation Learning (from HyperCBM).

    Builds a concept-level hypergraph where:
      - Nodes = concepts (not patches)
      - Hyperedges = concept co-occurrence patterns

    This enforces inter-concept consistency. For example, if a patch
    has high enhancement (c1) and high perfusion (c5), the concept
    hypergraph can learn that these should co-occur in active tumor.

    The concept hypergraph operates on the (num_concepts, D) matrix
    transposed from the (N, num_concepts) concept activation matrix.
    """

    def __init__(
        self,
        num_concepts: int = NUM_CONCEPTS,
        embed_dim: int = 32,
        num_heads: int = 2,
    ):
        super().__init__()
        self.num_concepts = num_concepts

        # Project concept activations to embedding space
        self.concept_embed = nn.Linear(1, embed_dim)

        # Self-attention for concept-to-concept dependencies
        # (lightweight: only 8 concepts, so full attention is fine)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

        # Project back to scalar concept values
        self.output_proj = nn.Linear(embed_dim, 1)

    def forward(self, concepts: torch.Tensor) -> torch.Tensor:
        """
        Refine concept predictions through inter-concept attention.

        Args:
            concepts: (N, num_concepts) raw concept predictions

        Returns:
            refined: (N, num_concepts) refined concept values
        """
        N = concepts.shape[0]

        # Reshape: treat each concept as a token
        # (N, C) → (N, C, 1) → (N, C, embed_dim)
        c_tokens = self.concept_embed(concepts.unsqueeze(-1))  # (N, C, D)

        # Self-attention across concepts (per node)
        attn_out, _ = self.attention(c_tokens, c_tokens, c_tokens)
        c_refined = self.norm(c_tokens + attn_out)  # (N, C, D)

        # Project back to scalar values
        refined = self.output_proj(c_refined).squeeze(-1)  # (N, C)

        return refined


class ConceptBottleneck(nn.Module):
    """
    Full Concept Bottleneck Module.

    Combines:
      1. ConceptPredictor: SHGNN embeddings → raw concept predictions
      2. HECRL: Inter-concept refinement
      3. Concept supervision loss (self-supervised MSE)

    The bottleneck ensures the downstream classifier can ONLY see
    concept activations, never raw SHGNN embeddings. This makes
    every prediction interpretable through concepts.
    """

    def __init__(
        self,
        embed_dim: int = SHEAF_HGNN_DIM,
        num_concepts: int = NUM_CONCEPTS,
        use_hecrl: bool = True,
        residual_bypass: bool = False,
    ):
        super().__init__()
        self.num_concepts = num_concepts
        self.use_hecrl = use_hecrl
        self.residual_bypass = residual_bypass

        # Concept prediction from SHGNN embeddings
        self.predictor = ConceptPredictor(embed_dim, num_concepts)

        # Inter-concept refinement (HECRL)
        if use_hecrl:
            self.hecrl = HECRL(num_concepts)

        # Optional: compute boundary complexity (c7) from graph neighborhood
        self.boundary_head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

        # Output dimension: concepts only (unless residual bypass enabled)
        if residual_bypass:
            self.output_dim = num_concepts + embed_dim
        else:
            self.output_dim = num_concepts

    def forward(
        self,
        node_embeddings: torch.Tensor,
        concept_targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            node_embeddings: (N, embed_dim) from SHGNN
            concept_targets: (N, num_concepts) precomputed concept GT (for loss)

        Returns:
            dict with:
                "concepts": (N, num_concepts) — final concept activations
                "concept_loss": scalar — supervision loss (if targets provided)
                "bottleneck_output": (N, output_dim) — input to classifier
                "concept_raw": (N, num_concepts) — pre-HECRL predictions
        """
        # ── Predict concepts from SHGNN embeddings ───────────────────
        concepts_raw = self.predictor(node_embeddings)  # (N, C)

        # ── Replace c7 (boundary complexity) with graph-aware version ─
        boundary = self.boundary_head(node_embeddings)  # (N, 1)
        concepts_raw = concepts_raw.clone()
        concepts_raw[:, 6] = boundary.squeeze(-1)  # c7 index = 6

        # ── HECRL refinement ─────────────────────────────────────────
        if self.use_hecrl:
            concepts_refined = self.hecrl(concepts_raw)
        else:
            concepts_refined = concepts_raw

        # ── Concept supervision loss ─────────────────────────────────
        concept_loss = torch.tensor(0.0, device=node_embeddings.device)
        if concept_targets is not None:
            # Supervise all concepts except c7 (boundary, which has no
            # precomputed target — it's learned from the graph)
            # Mask: supervise c1-c6, c8 (indices 0-5, 7)
            mask = torch.ones(self.num_concepts, device=node_embeddings.device)
            mask[6] = 0.0  # c7 is graph-learned, no GT target

            diff = (concepts_refined - concept_targets) ** 2  # (N, C)
            concept_loss = (diff * mask.unsqueeze(0)).mean()

        # ── Bottleneck output ────────────────────────────────────────
        if self.residual_bypass:
            bottleneck_output = torch.cat([concepts_refined, node_embeddings], dim=-1)
        else:
            bottleneck_output = concepts_refined

        return {
            "concepts": concepts_refined,
            "concept_loss": concept_loss,
            "bottleneck_output": bottleneck_output,
            "concept_raw": concepts_raw,
        }


class ConceptLoss(nn.Module):
    """
    Self-supervised concept prediction loss.

    Computes MSE between predicted and precomputed concept values.
    Handles missing modality concepts by masking.
    """

    def __init__(self, num_concepts: int = NUM_CONCEPTS):
        super().__init__()
        self.num_concepts = num_concepts

    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        modality_mask: Optional[Dict[str, bool]] = None,
    ) -> torch.Tensor:
        """
        Args:
            predicted: (N, C) predicted concepts
            target: (N, C) precomputed concept GT
            modality_mask: which modalities were present

        Returns:
            loss: scalar MSE loss (masked for missing modalities)
        """
        # Build concept-level mask based on modality availability
        concept_mask = torch.ones(self.num_concepts, device=predicted.device)
        concept_mask[6] = 0.0  # c7 is graph-learned

        if modality_mask is not None:
            if not modality_mask.get("DTI", True):
                concept_mask[3] = 0.0  # c4: DTI MD
                concept_mask[4] = 0.0  # c5: DTI FA
            if not modality_mask.get("Perfusion", True):
                pass  # Perfusion is used in c6 (heterogeneity) but not exclusively

        diff = (predicted - target) ** 2  # (N, C)
        masked_diff = diff * concept_mask.unsqueeze(0)  # broadcast mask

        # Mean over valid concepts and nodes
        n_valid = concept_mask.sum().clamp(min=1)
        return masked_diff.sum(dim=-1).mean() / n_valid
