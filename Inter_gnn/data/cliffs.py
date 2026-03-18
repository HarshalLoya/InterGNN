"""
Activity cliff detection.

Identifies pairs of structurally similar molecules with large differences
in biological activity (activity cliffs). These pairs are used for
supervision/diagnostics per the ACES-GNN methodology.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

logger = logging.getLogger(__name__)


def compute_fingerprints(
    smiles_list: List[str],
    fp_type: str = "morgan",
    radius: int = 2,
    n_bits: int = 2048,
) -> Tuple[List[DataStructs.ExplicitBitVect], List[int]]:
    """
    Compute molecular fingerprints for a list of SMILES.

    Args:
        smiles_list: List of SMILES strings.
        fp_type: Fingerprint type ('morgan', 'rdkit', 'maccs').
        radius: Morgan fingerprint radius (only for 'morgan').
        n_bits: Number of bits in the fingerprint vector.

    Returns:
        Tuple of:
            - fingerprints: List of RDKit fingerprint objects
            - valid_indices: Indices of successfully processed SMILES
    """
    fingerprints = []
    valid_indices = []

    for idx, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        try:
            if fp_type == "morgan":
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            elif fp_type == "rdkit":
                fp = Chem.RDKFingerprint(mol, fpSize=n_bits)
            elif fp_type == "maccs":
                fp = AllChem.GetMACCSKeysFingerprint(mol)
            else:
                raise ValueError(f"Unknown fingerprint type: {fp_type}")

            fingerprints.append(fp)
            valid_indices.append(idx)
        except Exception as e:
            logger.warning(f"Failed to compute fingerprint for index {idx}: {e}")

    return fingerprints, valid_indices


def compute_similarity_matrix(
    fingerprints: List[DataStructs.ExplicitBitVect],
    metric: str = "tanimoto",
) -> np.ndarray:
    """
    Compute pairwise similarity matrix from fingerprints.

    Uses bulk Tanimoto similarity for efficiency.

    Args:
        fingerprints: List of RDKit fingerprint objects.
        metric: Similarity metric ('tanimoto', 'dice', 'cosine').

    Returns:
        (N, N) symmetric similarity matrix.
    """
    n = len(fingerprints)
    sim_matrix = np.zeros((n, n), dtype=np.float32)

    if metric == "tanimoto":
        for i in range(n):
            # Bulk computation for row i
            sims = DataStructs.BulkTanimotoSimilarity(fingerprints[i], fingerprints)
            sim_matrix[i] = sims
    elif metric == "dice":
        for i in range(n):
            sims = DataStructs.BulkDiceSimilarity(fingerprints[i], fingerprints)
            sim_matrix[i] = sims
    elif metric == "cosine":
        for i in range(n):
            sims = DataStructs.BulkCosineSimilarity(fingerprints[i], fingerprints)
            sim_matrix[i] = sims
    else:
        raise ValueError(f"Unknown similarity metric: {metric}")

    return sim_matrix


def find_cliff_pairs(
    smiles_list: List[str],
    activities: List[float],
    sim_threshold: float = 0.9,
    act_threshold: float = 1.0,
    fp_type: str = "morgan",
    fp_radius: int = 2,
    fp_bits: int = 2048,
    metric: str = "tanimoto",
    max_pairs: Optional[int] = None,
) -> List[Dict]:
    """
    Identify activity cliff pairs from a set of molecules.

    Activity cliffs are pairs of molecules that are:
        1. Structurally similar (above sim_threshold)
        2. Have large activity differences (above act_threshold)

    Args:
        smiles_list: List of SMILES strings.
        activities: List of activity values (e.g., pIC50).
        sim_threshold: Minimum structural similarity for cliff candidate.
        act_threshold: Minimum activity difference for cliff classification.
        fp_type: Fingerprint type for similarity computation.
        fp_radius: Morgan fingerprint radius.
        fp_bits: Fingerprint bit length.
        metric: Similarity metric ('tanimoto', 'dice', 'cosine').
        max_pairs: Maximum number of cliff pairs to return (None for all).

    Returns:
        List of dicts, each containing:
            - 'idx_i': Index of first molecule
            - 'idx_j': Index of second molecule
            - 'smiles_i': SMILES of first molecule
            - 'smiles_j': SMILES of second molecule
            - 'similarity': Structural similarity score
            - 'activity_i': Activity of first molecule
            - 'activity_j': Activity of second molecule
            - 'activity_diff': Absolute activity difference
            - 'cliff_score': Combined cliff severity score
    """
    assert len(smiles_list) == len(activities), (
        f"SMILES list ({len(smiles_list)}) and activities ({len(activities)}) must have equal length"
    )

    # Compute fingerprints
    fingerprints, valid_indices = compute_fingerprints(
        smiles_list, fp_type, fp_radius, fp_bits
    )

    if len(fingerprints) < 2:
        logger.warning("Not enough valid molecules to find cliff pairs")
        return []

    # Build activity array for valid molecules
    valid_activities = np.array([activities[i] for i in valid_indices], dtype=np.float64)

    # Compute similarity matrix
    logger.info(f"Computing {len(fingerprints)}x{len(fingerprints)} similarity matrix...")
    sim_matrix = compute_similarity_matrix(fingerprints, metric)

    # Find cliff pairs
    cliff_pairs = []
    n = len(fingerprints)

    for i in range(n):
        for j in range(i + 1, n):
            sim = sim_matrix[i, j]
            if sim < sim_threshold:
                continue

            act_diff = abs(valid_activities[i] - valid_activities[j])
            if act_diff < act_threshold:
                continue

            # Cliff score: product of similarity and activity difference
            # Higher score = more surprising cliff
            cliff_score = sim * act_diff

            cliff_pairs.append({
                "idx_i": valid_indices[i],
                "idx_j": valid_indices[j],
                "smiles_i": smiles_list[valid_indices[i]],
                "smiles_j": smiles_list[valid_indices[j]],
                "similarity": float(sim),
                "activity_i": float(valid_activities[i]),
                "activity_j": float(valid_activities[j]),
                "activity_diff": float(act_diff),
                "cliff_score": float(cliff_score),
            })

    # Sort by cliff score (most severe first)
    cliff_pairs.sort(key=lambda x: x["cliff_score"], reverse=True)

    if max_pairs is not None:
        cliff_pairs = cliff_pairs[:max_pairs]

    logger.info(
        f"Found {len(cliff_pairs)} activity cliff pairs "
        f"(sim >= {sim_threshold}, act_diff >= {act_threshold})"
    )

    return cliff_pairs


def tag_cliff_molecules(
    smiles_list: List[str],
    activities: List[float],
    cliff_pairs: List[Dict],
) -> np.ndarray:
    """
    Create a binary tag array marking molecules involved in activity cliffs.

    Args:
        smiles_list: Full list of SMILES strings.
        activities: Full list of activity values.
        cliff_pairs: List of cliff pair dicts from `find_cliff_pairs`.

    Returns:
        Binary array of shape (N,) where 1 indicates cliff molecule.
    """
    tags = np.zeros(len(smiles_list), dtype=np.int64)

    for pair in cliff_pairs:
        tags[pair["idx_i"]] = 1
        tags[pair["idx_j"]] = 1

    n_cliff = int(tags.sum())
    logger.info(
        f"Tagged {n_cliff}/{len(smiles_list)} molecules as cliff-involved "
        f"({100 * n_cliff / len(smiles_list):.1f}%)"
    )

    return tags


def get_cliff_pair_indices(cliff_pairs: List[Dict]) -> List[Tuple[int, int]]:
    """Extract (i, j) index tuples from cliff pair dicts."""
    return [(p["idx_i"], p["idx_j"]) for p in cliff_pairs]


def compute_cliff_statistics(cliff_pairs: List[Dict]) -> Dict:
    """
    Compute summary statistics for detected activity cliffs.

    Returns:
        Dict with statistics including counts, mean/std of similarities
        and activity differences, and top cliff scores.
    """
    if not cliff_pairs:
        return {
            "num_pairs": 0,
            "num_unique_molecules": 0,
        }

    sims = [p["similarity"] for p in cliff_pairs]
    diffs = [p["activity_diff"] for p in cliff_pairs]
    scores = [p["cliff_score"] for p in cliff_pairs]

    unique_mols = set()
    for p in cliff_pairs:
        unique_mols.add(p["idx_i"])
        unique_mols.add(p["idx_j"])

    return {
        "num_pairs": len(cliff_pairs),
        "num_unique_molecules": len(unique_mols),
        "similarity_mean": float(np.mean(sims)),
        "similarity_std": float(np.std(sims)),
        "similarity_min": float(np.min(sims)),
        "similarity_max": float(np.max(sims)),
        "activity_diff_mean": float(np.mean(diffs)),
        "activity_diff_std": float(np.std(diffs)),
        "activity_diff_min": float(np.min(diffs)),
        "activity_diff_max": float(np.max(diffs)),
        "cliff_score_mean": float(np.mean(scores)),
        "cliff_score_max": float(np.max(scores)),
        "top_5_scores": sorted(scores, reverse=True)[:5],
    }
