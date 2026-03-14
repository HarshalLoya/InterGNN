"""
Dataset splitting strategies.

Provides scaffold splits, cold-target splits, temporal splits, and
random splits for robust model evaluation. Follows MoleculeNet and
DTA best practices to prevent data leakage.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

logger = logging.getLogger(__name__)


def _get_scaffold(smiles: str, generic: bool = False) -> Optional[str]:
    """
    Extract Bemis-Murcko scaffold SMILES from a molecule.

    Args:
        smiles: Input SMILES string.
        generic: If True, return generic scaffold (all atoms → C, all bonds → single).

    Returns:
        Scaffold SMILES string, or None on failure.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if generic:
            scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold)
        return Chem.MolToSmiles(scaffold, canonical=True)
    except Exception:
        return None


def scaffold_split(
    smiles_list: List[str],
    frac_train: float = 0.8,
    frac_val: float = 0.1,
    frac_test: float = 0.1,
    generic_scaffolds: bool = False,
    seed: int = 42,
    balanced: bool = True,
) -> Dict[str, List[int]]:
    """
    Split molecules by Bemis-Murcko scaffold to test generalization.

    Ensures all molecules with the same scaffold stay in the same split,
    preventing scaffold leakage between train/val/test sets.

    Args:
        smiles_list: List of SMILES strings.
        frac_train: Fraction for training set.
        frac_val: Fraction for validation set.
        frac_test: Fraction for test set.
        generic_scaffolds: Use generic scaffolds (ignores atom types).
        seed: Random seed for reproducibility.
        balanced: If True, distribute scaffold groups to balance split sizes.

    Returns:
        Dict with keys 'train', 'val', 'test', each mapping to a list of indices.
    """
    assert abs(frac_train + frac_val + frac_test - 1.0) < 1e-6, (
        "Fractions must sum to 1.0"
    )

    rng = np.random.RandomState(seed)

    # Group molecules by scaffold
    scaffold_to_indices: Dict[str, List[int]] = defaultdict(list)
    no_scaffold_indices = []

    for idx, smi in enumerate(smiles_list):
        scaffold = _get_scaffold(smi, generic=generic_scaffolds)
        if scaffold is not None:
            scaffold_to_indices[scaffold].append(idx)
        else:
            no_scaffold_indices.append(idx)

    # Sort scaffold groups by size (largest first for balanced allocation)
    scaffold_groups = list(scaffold_to_indices.values())

    if balanced:
        scaffold_groups.sort(key=len, reverse=True)
    else:
        rng.shuffle(scaffold_groups)

    # Allocate groups to splits
    n_total = len(smiles_list)
    n_train = int(n_total * frac_train)
    n_val = int(n_total * frac_val)

    train_indices = []
    val_indices = []
    test_indices = []

    for group in scaffold_groups:
        if len(train_indices) < n_train:
            train_indices.extend(group)
        elif len(val_indices) < n_val:
            val_indices.extend(group)
        else:
            test_indices.extend(group)

    # Distribute no-scaffold molecules randomly
    rng.shuffle(no_scaffold_indices)
    n_no_scaffold = len(no_scaffold_indices)
    n_ns_train = int(n_no_scaffold * frac_train)
    n_ns_val = int(n_no_scaffold * frac_val)

    train_indices.extend(no_scaffold_indices[:n_ns_train])
    val_indices.extend(no_scaffold_indices[n_ns_train : n_ns_train + n_ns_val])
    test_indices.extend(no_scaffold_indices[n_ns_train + n_ns_val :])

    logger.info(
        f"Scaffold split — Train: {len(train_indices)}, "
        f"Val: {len(val_indices)}, Test: {len(test_indices)} "
        f"(from {len(scaffold_to_indices)} unique scaffolds)"
    )

    return {
        "train": sorted(train_indices),
        "val": sorted(val_indices),
        "test": sorted(test_indices),
    }


def cold_target_split(
    target_ids: List[str],
    frac_train: float = 0.8,
    frac_val: float = 0.1,
    frac_test: float = 0.1,
    seed: int = 42,
) -> Dict[str, List[int]]:
    """
    Split drug-target pairs by target to test cold-target generalization.

    Ensures all interactions with the same target are in the same split.
    Useful for DTA tasks (Davis, KIBA) to test prediction on unseen proteins.

    Args:
        target_ids: List of target identifiers (one per sample).
        frac_train: Fraction for training.
        frac_val: Fraction for validation.
        frac_test: Fraction for test.
        seed: Random seed.

    Returns:
        Dict with 'train', 'val', 'test' index lists.
    """
    assert abs(frac_train + frac_val + frac_test - 1.0) < 1e-6

    rng = np.random.RandomState(seed)

    # Group samples by target
    target_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, tid in enumerate(target_ids):
        target_to_indices[tid].append(idx)

    # Shuffle targets
    unique_targets = list(target_to_indices.keys())
    rng.shuffle(unique_targets)

    n_targets = len(unique_targets)
    n_train_targets = int(n_targets * frac_train)
    n_val_targets = int(n_targets * frac_val)

    train_targets = unique_targets[:n_train_targets]
    val_targets = unique_targets[n_train_targets : n_train_targets + n_val_targets]
    test_targets = unique_targets[n_train_targets + n_val_targets :]

    train_indices = []
    val_indices = []
    test_indices = []

    for t in train_targets:
        train_indices.extend(target_to_indices[t])
    for t in val_targets:
        val_indices.extend(target_to_indices[t])
    for t in test_targets:
        test_indices.extend(target_to_indices[t])

    logger.info(
        f"Cold-target split — Train: {len(train_indices)} ({len(train_targets)} targets), "
        f"Val: {len(val_indices)} ({len(val_targets)} targets), "
        f"Test: {len(test_indices)} ({len(test_targets)} targets)"
    )

    return {
        "train": sorted(train_indices),
        "val": sorted(val_indices),
        "test": sorted(test_indices),
    }


def cold_drug_split(
    drug_ids: List[str],
    frac_train: float = 0.8,
    frac_val: float = 0.1,
    frac_test: float = 0.1,
    seed: int = 42,
) -> Dict[str, List[int]]:
    """
    Split drug-target pairs by drug to test cold-drug generalization.

    Same logic as cold_target_split but groups by drug identifier.

    Args:
        drug_ids: List of drug identifiers (one per sample).
        frac_train, frac_val, frac_test: Split fractions.
        seed: Random seed.

    Returns:
        Dict with 'train', 'val', 'test' index lists.
    """
    return cold_target_split(drug_ids, frac_train, frac_val, frac_test, seed)


def temporal_split(
    timestamps: List[Union[int, float, str]],
    frac_train: float = 0.8,
    frac_val: float = 0.1,
    frac_test: float = 0.1,
) -> Dict[str, List[int]]:
    """
    Chronological split based on timestamps.

    Ensures test data is temporally after training data, simulating
    prospective evaluation.

    Args:
        timestamps: List of timestamps (sortable values) per sample.
        frac_train, frac_val, frac_test: Split fractions.

    Returns:
        Dict with 'train', 'val', 'test' index lists.
    """
    assert abs(frac_train + frac_val + frac_test - 1.0) < 1e-6

    # Sort by timestamp
    sorted_indices = sorted(range(len(timestamps)), key=lambda i: timestamps[i])

    n_total = len(sorted_indices)
    n_train = int(n_total * frac_train)
    n_val = int(n_total * frac_val)

    train_indices = sorted_indices[:n_train]
    val_indices = sorted_indices[n_train : n_train + n_val]
    test_indices = sorted_indices[n_train + n_val :]

    logger.info(
        f"Temporal split — Train: {len(train_indices)}, "
        f"Val: {len(val_indices)}, Test: {len(test_indices)}"
    )

    return {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }


def random_split(
    n_samples: int,
    frac_train: float = 0.8,
    frac_val: float = 0.1,
    frac_test: float = 0.1,
    seed: int = 42,
    stratify_labels: Optional[List[int]] = None,
) -> Dict[str, List[int]]:
    """
    Random split with optional stratification.

    Args:
        n_samples: Total number of samples.
        frac_train, frac_val, frac_test: Split fractions.
        seed: Random seed.
        stratify_labels: If provided, maintain class distribution across splits.

    Returns:
        Dict with 'train', 'val', 'test' index lists.
    """
    assert abs(frac_train + frac_val + frac_test - 1.0) < 1e-6

    if stratify_labels is not None:
        from sklearn.model_selection import train_test_split

        indices = list(range(n_samples))

        # First split: train+val vs test
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=frac_test,
            random_state=seed,
            stratify=[stratify_labels[i] for i in indices],
        )

        # Second split: train vs val
        relative_val_frac = frac_val / (frac_train + frac_val)
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=relative_val_frac,
            random_state=seed,
            stratify=[stratify_labels[i] for i in train_val_idx],
        )

        return {
            "train": sorted(train_idx),
            "val": sorted(val_idx),
            "test": sorted(test_idx),
        }
    else:
        rng = np.random.RandomState(seed)
        indices = np.arange(n_samples)
        rng.shuffle(indices)

        n_train = int(n_samples * frac_train)
        n_val = int(n_samples * frac_val)

        return {
            "train": sorted(indices[:n_train].tolist()),
            "val": sorted(indices[n_train : n_train + n_val].tolist()),
            "test": sorted(indices[n_train + n_val :].tolist()),
        }


def k_fold_split(
    n_samples: int,
    n_folds: int = 5,
    seed: int = 42,
    stratify_labels: Optional[List[int]] = None,
) -> List[Dict[str, List[int]]]:
    """
    K-fold cross-validation split.

    Args:
        n_samples: Total number of samples.
        n_folds: Number of folds.
        seed: Random seed.
        stratify_labels: If provided, maintain class distribution.

    Returns:
        List of dicts, one per fold, with 'train' and 'val' index lists.
    """
    if stratify_labels is not None:
        from sklearn.model_selection import StratifiedKFold
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = []
        for train_idx, val_idx in kf.split(range(n_samples), stratify_labels):
            splits.append({
                "train": sorted(train_idx.tolist()),
                "val": sorted(val_idx.tolist()),
            })
    else:
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = []
        for train_idx, val_idx in kf.split(range(n_samples)):
            splits.append({
                "train": sorted(train_idx.tolist()),
                "val": sorted(val_idx.tolist()),
            })

    return splits
