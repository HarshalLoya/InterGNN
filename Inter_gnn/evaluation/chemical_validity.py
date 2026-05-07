"""
Chemical validity metrics for generated explanations.

Checks whether explanation substructures (motifs, counterfactuals)
correspond to chemically valid molecules.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any

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


# ── Known toxicophore / pharmacophore SMARTS ──
TOXICOPHORE_SMARTS = {
    "nitroaromatic": "[$(c1ccccc1[N+](=O)[O-]),$(c1ccccc1N(=O)=O)]",
    "aromatic_amine": "[NX3;H2,H1;!$(NC=O)]c1ccccc1",
    "epoxide": "C1OC1",
    "michael_acceptor": "[CX3]=[CX3][CX3]=[OX1]",
    "acyl_halide": "[CX3](=[OX1])[F,Cl,Br,I]",
    "aldehyde": "[CX3H1](=O)[#6]",
    "sulfonyl_halide": "[SX4](=[OX1])(=[OX1])[F,Cl,Br,I]",
    "halogenated_aromatic": "c1ccccc1[F,Cl,Br,I]",
    "phenol": "c1ccc(O)cc1",
    "quinone": "O=C1C=CC(=O)C=C1",
    "hydrazine": "[NX3][NX3]",
    "isocyanate": "[NX2]=C=O",
}


def toxicophore_recovery_score(
    smiles_list: List[str],
    atom_importance_list: List[List[float]],
    top_k_fraction: float = 0.3,
    smarts_dict: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Measure how well the model's most important atoms overlap with known
    toxicophoric / pharmacophoric substructures.

    For each molecule that contains a toxicophore pattern, we compute the
    fraction of pattern-matched atoms that fall within the model's top-K
    most important atoms (precision) and the fraction of top-K atoms that
    are pattern-matched (recall).

    Args:
        smiles_list: SMILES of molecules.
        atom_importance_list: Per-atom importance scores (one list per molecule).
        top_k_fraction: Fraction of atoms to consider as "important".
        smarts_dict: Dict of {name: SMARTS} patterns. Defaults to TOXICOPHORE_SMARTS.

    Returns:
        Dict with per-pattern recovery rates and aggregate statistics.
    """
    if not HAS_RDKIT:
        return {"error": "RDKit not available"}

    patterns = smarts_dict or TOXICOPHORE_SMARTS
    results = {"per_pattern": {}, "molecules_with_toxicophores": 0, "total_molecules": len(smiles_list)}
    all_precisions = []
    all_recalls = []

    for smi, importance in zip(smiles_list, atom_importance_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        n_atoms = mol.GetNumAtoms()
        if len(importance) < n_atoms:
            continue

        # Determine top-K important atom indices
        k = max(1, int(n_atoms * top_k_fraction))
        imp_arr = np.array(importance[:n_atoms])
        top_k_indices = set(np.argsort(imp_arr)[-k:].tolist())

        mol_has_toxicophore = False
        for pat_name, smarts in patterns.items():
            pat = Chem.MolFromSmarts(smarts)
            if pat is None:
                continue
            matches = mol.GetSubstructMatches(pat)
            if not matches:
                continue

            mol_has_toxicophore = True
            matched_atoms = set()
            for match in matches:
                matched_atoms.update(match)

            # Precision: of matched atoms, how many are in top-K?
            if matched_atoms:
                precision = len(matched_atoms & top_k_indices) / len(matched_atoms)
            else:
                precision = 0.0
            # Recall: of top-K atoms, how many are matched?
            recall = len(matched_atoms & top_k_indices) / len(top_k_indices) if top_k_indices else 0.0

            all_precisions.append(precision)
            all_recalls.append(recall)

            if pat_name not in results["per_pattern"]:
                results["per_pattern"][pat_name] = {"precisions": [], "recalls": [], "count": 0}
            results["per_pattern"][pat_name]["precisions"].append(precision)
            results["per_pattern"][pat_name]["recalls"].append(recall)
            results["per_pattern"][pat_name]["count"] += 1

        if mol_has_toxicophore:
            results["molecules_with_toxicophores"] += 1

    # Aggregate per-pattern
    for pat_name, data in results["per_pattern"].items():
        data["mean_precision"] = float(np.mean(data["precisions"])) if data["precisions"] else 0.0
        data["mean_recall"] = float(np.mean(data["recalls"])) if data["recalls"] else 0.0
        del data["precisions"]
        del data["recalls"]

    results["overall_mean_precision"] = float(np.mean(all_precisions)) if all_precisions else 0.0
    results["overall_mean_recall"] = float(np.mean(all_recalls)) if all_recalls else 0.0
    results["num_evaluated_pairs"] = len(all_precisions)

    return results
