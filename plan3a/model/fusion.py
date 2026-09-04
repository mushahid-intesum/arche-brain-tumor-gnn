"""
Multimodal Fusion Module for Plan 3a (MRePath-inspired).

Fuses imaging hypergraph features with clinical/molecular features using:
  1. Dynamic Weighting: Mono-confidence + Holo-confidence (MRePath Eq. 7-9)
  2. Interactive Alignment Fusion: Bidirectional cross-attention

MRePath Analogy:
  - "Pathology" modality → Graph-level imaging embedding (from SHGNN + pooling)
  - "Genomics" modality → Clinical feature vector (from clinical.py)

The dynamic weighting learns to balance imaging vs clinical contributions
per-patient, preventing one modality from dominating.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from plan3a.config import EMBED_DIM, SHEAF_HGNN_DIM
from plan3a.data.clinical import get_feature_dim


class ClinicalEncoder(nn.Module):
    """
    Encode clinical features into the same latent space as imaging.

    Input: (B, clinical_dim) raw clinical features with missingness flags
    Output: (B, embed_dim) clinical embedding
    """

    def __init__(self, clinical_dim: int = None, embed_dim: int = SHEAF_HGNN_DIM):
        super().__init__()
        if clinical_dim is None:
            clinical_dim = get_feature_dim()

        self.encoder = nn.Sequential(
            nn.Linear(clinical_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, clinical_dim) → (B, embed_dim)"""
        return self.encoder(x)


class DynamicWeighting(nn.Module):
    """
    Dynamic Modality Rebalancing (MRePath Eq. 7-9).

    Computes per-sample weights for imaging and clinical modalities:
      - Mono-confidence: each modality's standalone reliability
      - Holo-confidence: cross-modal interaction strength
      - Final weights: softmax(mono + holo) for each modality

    This prevents modality collapse — if clinical data is heavily missing
    for a patient, the model automatically upweights imaging.
    """

    def __init__(self, embed_dim: int = SHEAF_HGNN_DIM):
        super().__init__()
        # Mono-confidence projections (Φ_p, Φ_g in MRePath)
        self.img_confidence = nn.Linear(embed_dim, 1)
        self.clin_confidence = nn.Linear(embed_dim, 1)

    def forward(
        self,
        img_embed: torch.Tensor,
        clin_embed: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            img_embed: (B, embed_dim) imaging embedding
            clin_embed: (B, embed_dim) clinical embedding

        Returns:
            w_img: (B, 1) imaging weight
            w_clin: (B, 1) clinical weight
        """
        # Mono-confidence: standalone reliability of each modality
        w_img_mono = torch.sigmoid(self.img_confidence(img_embed))    # (B, 1)
        w_clin_mono = torch.sigmoid(self.clin_confidence(clin_embed))  # (B, 1)

        eps = 1e-8

        # Holo-confidence: cross-modal interaction strength
        # w_p^h = log(w_p^m) / log(w_p^m · w_g^m)
        product = (w_img_mono * w_clin_mono).clamp(min=eps)
        w_img_holo = torch.log(w_img_mono.clamp(min=eps)) / (torch.log(product) + eps)
        w_clin_holo = torch.log(w_clin_mono.clamp(min=eps)) / (torch.log(product) + eps)

        # Clamp to prevent NaN from log operations
        w_img_holo = w_img_holo.clamp(-5, 5)
        w_clin_holo = w_clin_holo.clamp(-5, 5)

        # Final weights: softmax over modalities
        scores = torch.cat([
            w_img_mono + w_img_holo,
            w_clin_mono + w_clin_holo,
        ], dim=-1)  # (B, 2)
        weights = F.softmax(scores, dim=-1)  # (B, 2)

        w_img = weights[:, 0:1]   # (B, 1)
        w_clin = weights[:, 1:2]  # (B, 1)

        return w_img, w_clin


class InteractiveAlignmentFusion(nn.Module):
    """
    Interactive Alignment Fusion via cross-attention (MRePath).

    Bidirectional cross-attention between imaging and clinical embeddings:
      1. Clinical-guided: clinical queries imaging → selects relevant features
      2. Image-guided: imaging queries clinical → contextualizes clinical meaning

    Output: fused embedding combining both modalities.
    """

    def __init__(self, embed_dim: int = SHEAF_HGNN_DIM, num_heads: int = 4):
        super().__init__()

        # Clinical-guided cross-attention (clinical queries imaging)
        self.clin_to_img_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            dropout=0.1, batch_first=True,
        )

        # Image-guided cross-attention (imaging queries clinical)
        self.img_to_clin_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            dropout=0.1, batch_first=True,
        )

        # Layer norms for residual connections
        self.norm_img = nn.LayerNorm(embed_dim)
        self.norm_clin = nn.LayerNorm(embed_dim)

        # Final fusion projection
        self.fusion_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(
        self,
        img_embed: torch.Tensor,
        clin_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            img_embed: (B, embed_dim) imaging embedding
            clin_embed: (B, embed_dim) clinical embedding

        Returns:
            fused: (B, embed_dim) fused multi-modal embedding
        """
        # Reshape for attention: (B, 1, D) — single token per modality
        img_q = img_embed.unsqueeze(1)    # (B, 1, D)
        clin_q = clin_embed.unsqueeze(1)  # (B, 1, D)

        # Clinical-guided: clinical queries select imaging features
        img_context, _ = self.clin_to_img_attn(clin_q, img_q, img_q)
        img_fused = self.norm_img(img_embed + img_context.squeeze(1))

        # Image-guided: imaging queries contextualize clinical
        clin_context, _ = self.img_to_clin_attn(img_q, clin_q, clin_q)
        clin_fused = self.norm_clin(clin_embed + clin_context.squeeze(1))

        # Concatenate and project
        fused = self.fusion_proj(torch.cat([img_fused, clin_fused], dim=-1))

        return fused


class MultiModalFusion(nn.Module):
    """
    Complete Multimodal Fusion Module.

    Combines:
      1. ClinicalEncoder: raw features → embedding
      2. DynamicWeighting: per-sample modality balancing
      3. InteractiveAlignmentFusion: cross-attention fusion

    Input: imaging embedding + clinical features
    Output: fused embedding for survival prediction
    """

    def __init__(
        self,
        embed_dim: int = SHEAF_HGNN_DIM,
        clinical_dim: int = None,
    ):
        super().__init__()
        self.clinical_encoder = ClinicalEncoder(clinical_dim, embed_dim)
        self.dynamic_weighting = DynamicWeighting(embed_dim)
        self.interactive_fusion = InteractiveAlignmentFusion(embed_dim)

    def forward(
        self,
        img_embed: torch.Tensor,
        clinical_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            img_embed: (B, embed_dim) graph-level imaging embedding
            clinical_features: (B, clinical_dim) raw clinical features

        Returns:
            dict with:
                "fused": (B, embed_dim) fused embedding
                "w_img": (B, 1) imaging weight
                "w_clin": (B, 1) clinical weight
                "clin_embed": (B, embed_dim) clinical embedding
        """
        # Encode clinical features
        clin_embed = self.clinical_encoder(clinical_features)  # (B, D)

        # Dynamic weighting
        w_img, w_clin = self.dynamic_weighting(img_embed, clin_embed)

        # Apply weights
        img_weighted = img_embed * w_img
        clin_weighted = clin_embed * w_clin

        # Interactive alignment fusion
        fused = self.interactive_fusion(img_weighted, clin_weighted)

        return {
            "fused": fused,
            "w_img": w_img,
            "w_clin": w_clin,
            "clin_embed": clin_embed,
        }
