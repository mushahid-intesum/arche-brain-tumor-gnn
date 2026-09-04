"""
PyTorch Dataset for Plan 3a.

Loads preprocessed .pt files and builds hypergraphs on-the-fly.
Handles the variable-size nature of patient graphs (different
numbers of patches/nodes per patient).
"""
import os
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from plan3a.config import PROCESSED_DIR
from plan3a.data.hypergraph import build_patient_hypergraph


class Plan3aDataset(Dataset):
    """
    Dataset yielding preprocessed patient hypergraphs.

    Each item is a dict containing node features, hypergraph incidence,
    concept targets, clinical features, and survival labels.

    Since each patient has a different number of patches/nodes,
    this cannot be batched with standard DataLoader. Use batch_size=1
    or a custom collate function.
    """

    def __init__(
        self,
        processed_dir: str = None,
        patient_ids: Optional[List[str]] = None,
        build_hypergraph: bool = True,
        precompute: bool = False,
    ):
        """
        Args:
            processed_dir: path to directory with .pt files
            patient_ids: list of patient IDs to include (for train/val splits)
            build_hypergraph: if True, build hypergraph on-the-fly
            precompute: if True, precompute and cache all hypergraphs in memory
        """
        if processed_dir is None:
            processed_dir = str(PROCESSED_DIR)

        self.processed_dir = processed_dir
        self.build_hypergraph = build_hypergraph

        # Discover available patients
        all_files = sorted([
            f for f in os.listdir(processed_dir)
            if f.endswith(".pt") and f.startswith("UPENN")
        ])

        if patient_ids is not None:
            self.files = [f for f in all_files if f.replace(".pt", "") in patient_ids]
        else:
            self.files = all_files

        self.patient_ids = [f.replace(".pt", "") for f in self.files]

        # Optional precomputation
        self.cache = {}
        if precompute:
            print(f"Precomputing hypergraphs for {len(self.files)} patients...")
            for i, f in enumerate(self.files):
                data = torch.load(
                    os.path.join(processed_dir, f), weights_only=False
                )
                if build_hypergraph:
                    data = build_patient_hypergraph(data)
                self.cache[i] = data
                if (i + 1) % 10 == 0:
                    print(f"  {i+1}/{len(self.files)}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict:
        if idx in self.cache:
            return self.cache[idx]

        filepath = os.path.join(self.processed_dir, self.files[idx])
        data = torch.load(filepath, weights_only=False)

        if self.build_hypergraph:
            data = build_patient_hypergraph(data)

        return data


def get_kfold_splits(
    patient_ids: List[str],
    num_folds: int = 5,
    seed: int = 42,
) -> List[Dict[str, List[str]]]:
    """
    Generate K-fold cross-validation splits.

    Returns list of dicts with 'train' and 'val' patient ID lists.
    """
    import numpy as np
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(patient_ids))
    fold_sizes = np.full(num_folds, len(patient_ids) // num_folds, dtype=int)
    fold_sizes[:len(patient_ids) % num_folds] += 1

    splits = []
    current = 0
    folds = []
    for size in fold_sizes:
        folds.append(indices[current:current + size].tolist())
        current += size

    for i in range(num_folds):
        val_ids = [patient_ids[j] for j in folds[i]]
        train_ids = [patient_ids[j] for fold_idx, fold in enumerate(folds)
                     if fold_idx != i for j in fold]
        splits.append({"train": train_ids, "val": val_ids})

    return splits
