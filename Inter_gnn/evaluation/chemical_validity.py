"""
Chemical validity metrics for generated explanations.

Checks whether explanation substructures (motifs, counterfactuals)
correspond to chemically valid molecules.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


def valence_check(smiles: str) -> bool:
    """Check if a SMILES string encodes a molecule with valid valences."""
    if not HAS_RDKIT:
        return True
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return False
    try:
        Chem.SanitizeMol(mol)
        return True
    except Exception:
        return False


def smarts_match_rate(
    smiles_list: List[str],
    smarts_patterns: List[str],
) -> Dict[str, float]:
    """
    Compute match rate of SMARTS patterns across a set of molecules.

    Args:
        smiles_list: List of SMILES strings.
        smarts_patterns: List of SMARTS patterns to check.

    Returns:
        Dict with per-pattern match rates and overall statistics.
    """
    if not HAS_RDKIT:
        return {"error": "RDKit not available"}

    results = {}
    overall_matches = 0
    overall_total = 0

    for smarts in smarts_patterns:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            results[smarts] = 0.0
            continue

        count = 0
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol and mol.HasSubstructMatch(pattern):
                count += 1

        rate = count / len(smiles_list) if smiles_list else 0.0
        results[smarts] = rate
        overall_matches += count
        overall_total += len(smiles_list)

    results["overall_match_rate"] = overall_matches / max(overall_total, 1)
    return results


def explanation_validity_report(
    explanation_smiles: List[str],
    original_smiles: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Comprehensive validity check for generated explanation molecules.

    Args:
        explanation_smiles: SMILES of explanation substructures.
        original_smiles: Optional original SMILES for comparison.

    Returns:
        Dict with validity rates, property distributions.
    """
    if not HAS_RDKIT:
        return {"error": "RDKit not available"}

    valid_count = 0
    valid_mols = []

    for smi in explanation_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            try:
                Chem.SanitizeMol(mol)
                valid_count += 1
                valid_mols.append(mol)
            except Exception:
                pass

    validity_rate = valid_count / len(explanation_smiles) if explanation_smiles else 0.0

    # Property distributions of valid explanations
    mw_values = [Descriptors.MolWt(m) for m in valid_mols]
    logp_values = [Descriptors.MolLogP(m) for m in valid_mols]
    ha_values = [m.GetNumHeavyAtoms() for m in valid_mols]

    report = {
        "validity_rate": validity_rate,
        "num_valid": valid_count,
        "num_total": len(explanation_smiles),
        "mean_molecular_weight": float(np.mean(mw_values)) if mw_values else 0.0,
        "mean_logp": float(np.mean(logp_values)) if logp_values else 0.0,
        "mean_heavy_atoms": float(np.mean(ha_values)) if ha_values else 0.0,
    }

    # Compare with original molecules if provided
    if original_smiles:
        orig_mols = [Chem.MolFromSmiles(s) for s in original_smiles if Chem.MolFromSmiles(s)]
        orig_mw = [Descriptors.MolWt(m) for m in orig_mols]
        orig_logp = [Descriptors.MolLogP(m) for m in orig_mols]

        if orig_mw and mw_values:
            report["mw_shift"] = float(np.mean(mw_values) - np.mean(orig_mw))
            report["logp_shift"] = float(np.mean(logp_values) - np.mean(orig_logp))

    return report
