"""
Concept Whitening validation utilities.

Provides tools to verify that learned concept axes are semantically
aligned with SMARTS-based chemical concepts by computing:
  - Per-concept top-K activating molecules
  - Concept purity scores
  - Concept-concept correlation heatmaps
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from rdkit import Chem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

logger = logging.getLogger(__name__)

# Default SMARTS concepts used in InterGNN concept whitening
DEFAULT_CONCEPT_SMARTS = [
    "c1ccccc1",            # benzene ring
    "[OH]",                # hydroxyl
    "[NH2]",               # primary amine
    "C(=O)O",              # carboxylic acid
    "C(=O)N",              # amide
    "[#7]",                # any nitrogen
    "[#8]",                # any oxygen
    "[#16]",               # any sulfur
    "[F,Cl,Br,I]",         # halogen
    "C=O",                 # carbonyl
    "C#N",                 # nitrile
    "c1ccncc1",            # pyridine
    "C1CCNCC1",            # piperidine
    "c1cc[nH]c1",          # pyrrole
    "c1cnc2ccccc2n1",      # quinazoline
    "C1CCCCC1",            # cyclohexane
    "C1CCC1",              # cyclobutane
    "C1CC1",               # cyclopropane
    "[N+](=O)[O-]",        # nitro group
    "c1ccc2[nH]ccc2c1",    # indole
    "c1ccc2ccccc2c1",      # naphthalene
    "OC(=O)C",             # ester
    "S(=O)(=O)",           # sulfonyl
    "c1ccsc1",             # thiophene
    "c1ccocc1",            # furan (loose)
    "C=C",                 # alkene
    "[#6]-[#8]-[#6]",      # ether
    "c1ccc(N)cc1",         # aniline
    "C(F)(F)F",            # trifluoromethyl
    "c1ccoc1",             # furan
]


def compute_concept_ground_truth(smiles_list: List[str], smarts_list: Optional[List[str]] = None) -> np.ndarray:
    """
    Compute binary ground-truth concept matrix from SMARTS patterns.

    Args:
        smiles_list: List of SMILES strings.
        smarts_list: List of SMARTS patterns. Defaults to DEFAULT_CONCEPT_SMARTS.

    Returns:
        np.ndarray of shape (N_molecules, N_concepts) with binary values.
    """
    if not HAS_RDKIT:
        raise ImportError("RDKit is required for concept validation")

    patterns = smarts_list or DEFAULT_CONCEPT_SMARTS
    compiled = []
    for s in patterns:
        p = Chem.MolFromSmarts(s)
        compiled.append(p)

    matrix = np.zeros((len(smiles_list), len(patterns)), dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        for j, pat in enumerate(compiled):
            if pat is not None and mol.HasSubstructMatch(pat):
                matrix[i, j] = 1.0

    return matrix


def concept_axis_purity(
    concept_activations: np.ndarray,
    concept_ground_truth: np.ndarray,
) -> Dict[str, Any]:
    """
    Measure how well each concept axis in the model aligns with
    its corresponding ground-truth concept.

    Purity score for axis j: correlation between activation on axis j
    and ground-truth label for concept j.

    Args:
        concept_activations: (N, C) activations from the concept whitening layer.
        concept_ground_truth: (N, C) binary ground-truth concept presence.

    Returns:
        Dict with per-axis purity and overall statistics.
    """
    n_concepts = min(concept_activations.shape[1], concept_ground_truth.shape[1])
    purities = []
    per_axis = {}

    for j in range(n_concepts):
        act = concept_activations[:, j]
        gt = concept_ground_truth[:, j]
        # Purity = Pearson correlation between activation and ground truth
        if np.std(act) < 1e-8 or np.std(gt) < 1e-8:
            corr = 0.0
        else:
            corr = float(np.corrcoef(act, gt)[0, 1])
            if np.isnan(corr):
                corr = 0.0
        purities.append(corr)
        per_axis[f"concept_{j}"] = corr

    return {
        "per_axis_purity": per_axis,
        "mean_purity": float(np.mean(purities)),
        "max_purity": float(np.max(purities)) if purities else 0.0,
        "min_purity": float(np.min(purities)) if purities else 0.0,
    }


def concept_correlation_matrix(concept_activations: np.ndarray) -> np.ndarray:
    """
    Compute pairwise correlation matrix between concept axes.

    A good concept whitening layer should produce low off-diagonal
    correlations, indicating independent concept representations.

    Args:
        concept_activations: (N, C) activations from concept whitening.

    Returns:
        (C, C) correlation matrix.
    """
    # Handle constant columns
    stds = np.std(concept_activations, axis=0)
    valid = stds > 1e-8
    if not valid.any():
        return np.zeros((concept_activations.shape[1], concept_activations.shape[1]))

    corr = np.corrcoef(concept_activations.T)
    corr = np.nan_to_num(corr, nan=0.0)
    return corr


def top_activating_molecules(
    concept_activations: np.ndarray,
    smiles_list: List[str],
    top_k: int = 5,
) -> Dict[int, List[Tuple[str, float]]]:
    """
    For each concept axis, find the top-K molecules with highest activation.

    Args:
        concept_activations: (N, C) matrix of concept activations.
        smiles_list: List of SMILES strings (length N).
        top_k: Number of top molecules to return per concept.

    Returns:
        Dict mapping concept index to list of (smiles, activation) tuples.
    """
    n_concepts = concept_activations.shape[1]
    result = {}

    for j in range(n_concepts):
        acts = concept_activations[:, j]
        top_indices = np.argsort(acts)[-top_k:][::-1]
        top_mols = []
        for idx in top_indices:
            if idx < len(smiles_list):
                top_mols.append((smiles_list[idx], float(acts[idx])))
        result[j] = top_mols

    return result


def generate_concept_validation_report(
    concept_activations: np.ndarray,
    smiles_list: List[str],
    smarts_list: Optional[List[str]] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Full concept whitening validation report.

    Computes purity scores, correlation heatmap data, and top activating
    molecules for each concept.

    Args:
        concept_activations: (N, C) activations from concept whitening.
        smiles_list: List of SMILES strings.
        smarts_list: Optional SMARTS patterns for ground truth.
        top_k: Number of top molecules per concept.

    Returns:
        Dict with all validation metrics.
    """
    gt = compute_concept_ground_truth(smiles_list, smarts_list)
    purity = concept_axis_purity(concept_activations, gt)
    corr_matrix = concept_correlation_matrix(concept_activations)
    top_mols = top_activating_molecules(concept_activations, smiles_list, top_k)

    # Off-diagonal correlation statistics
    n = corr_matrix.shape[0]
    if n > 1:
        mask = ~np.eye(n, dtype=bool)
        off_diag = np.abs(corr_matrix[mask])
        off_diag_stats = {
            "mean_abs_off_diagonal": float(np.mean(off_diag)),
            "max_abs_off_diagonal": float(np.max(off_diag)),
        }
    else:
        off_diag_stats = {"mean_abs_off_diagonal": 0.0, "max_abs_off_diagonal": 0.0}

    return {
        "purity": purity,
        "correlation_stats": off_diag_stats,
        "correlation_matrix": corr_matrix.tolist(),
        "top_activating_molecules": {str(k): v for k, v in top_mols.items()},
        "num_concepts": concept_activations.shape[1],
        "num_molecules": len(smiles_list),
    }

