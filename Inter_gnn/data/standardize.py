"""
Molecule standardization pipeline.

Provides tautomer canonicalization, charge normalization, stereochemistry
preservation, and duplicate removal with activity aggregation.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, inchi
from rdkit.Chem.MolStandardize import rdMolStandardize

logger = logging.getLogger(__name__)

# Suppress RDKit warnings during batch processing
RDLogger.DisableLog("rdApp.*")


def _get_largest_fragment(mol: Chem.Mol) -> Chem.Mol:
    """Return the largest fragment from a molecule (removes salts/counterions)."""
    fragment_chooser = rdMolStandardize.LargestFragmentChooser()
    return fragment_chooser.choose(mol)


def _uncharge_molecule(mol: Chem.Mol) -> Chem.Mol:
    """Neutralize charges where chemically reasonable."""
    uncharger = rdMolStandardize.Uncharger()
    return uncharger.uncharge(mol)


def _canonicalize_tautomer(mol: Chem.Mol) -> Chem.Mol:
    """Return the canonical tautomer of a molecule."""
    enumerator = rdMolStandardize.TautomerEnumerator()
    return enumerator.Canonicalize(mol)


def _normalize_molecule(mol: Chem.Mol) -> Chem.Mol:
    """Apply RDKit normalization transforms (functional group standardization)."""
    normalizer = rdMolStandardize.Normalizer()
    return normalizer.normalize(mol)


def standardize_mol(
    smiles: str,
    remove_stereo: bool = False,
    canonicalize_tautomers: bool = True,
    neutralize_charges: bool = True,
    keep_largest_fragment: bool = True,
) -> Optional[str]:
    """
    Standardize a single molecule from SMILES string.

    Pipeline:
        1. Parse SMILES → RDKit Mol
        2. Keep largest fragment (remove salts)
        3. Normalize functional groups
        4. Neutralize charges
        5. Canonicalize tautomers
        6. Optionally remove stereochemistry
        7. Return canonical SMILES

    Args:
        smiles: Input SMILES string.
        remove_stereo: If True, strip stereochemistry information.
        canonicalize_tautomers: If True, convert to canonical tautomer.
        neutralize_charges: If True, neutralize charges where possible.
        keep_largest_fragment: If True, remove salts/counterions.

    Returns:
        Standardized canonical SMILES, or None if parsing fails.
    """
    if not smiles or not isinstance(smiles, str):
        return None

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Failed to parse SMILES: {smiles}")
            return None

        # Step 1: Keep largest fragment
        if keep_largest_fragment:
            mol = _get_largest_fragment(mol)

        # Step 2: Normalize functional groups
        mol = _normalize_molecule(mol)

        # Step 3: Neutralize charges
        if neutralize_charges:
            mol = _uncharge_molecule(mol)

        # Step 4: Canonicalize tautomers
        if canonicalize_tautomers:
            mol = _canonicalize_tautomer(mol)

        # Step 5: Optionally strip stereochemistry
        if remove_stereo:
            Chem.RemoveStereochemistry(mol)

        # Generate canonical SMILES
        canon_smiles = Chem.MolToSmiles(mol, canonical=True)
        return canon_smiles

    except Exception as e:
        logger.warning(f"Standardization failed for {smiles}: {e}")
        return None


def remove_duplicates(
    df: pd.DataFrame,
    smiles_col: str = "smiles",
    activity_col: Optional[str] = None,
    aggregation: str = "median",
) -> pd.DataFrame:
    """
    Remove duplicate molecules (by InChIKey) and aggregate activity values.

    Args:
        df: DataFrame with at least a SMILES column.
        smiles_col: Name of the column containing SMILES strings.
        activity_col: Name of the activity/label column to aggregate.
            If None, simply drops duplicates.
        aggregation: Aggregation method for duplicate activities.
            One of 'median', 'mean', 'max', 'min'.

    Returns:
        De-duplicated DataFrame with standardized SMILES and InChIKey.
    """
    df = df.copy()

    # Compute InChIKey for each molecule
    inchi_keys = []
    for smi in df[smiles_col]:
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is not None:
            try:
                ik = inchi.MolToInchiKey(inchi.MolFromMolToInchi(mol))
            except Exception:
                ik = Chem.MolToSmiles(mol, canonical=True)
        else:
            ik = None
        inchi_keys.append(ik)

    df["_inchi_key"] = inchi_keys

    # Drop rows with failed InChIKey generation
    df = df.dropna(subset=["_inchi_key"])

    if activity_col is not None and activity_col in df.columns:
        agg_funcs = {
            "median": "median",
            "mean": "mean",
            "max": "max",
            "min": "min",
        }
        agg_func = agg_funcs.get(aggregation, "median")

        # Group by InChIKey, aggregate activity, keep first SMILES
        grouped = df.groupby("_inchi_key").agg(
            {smiles_col: "first", activity_col: agg_func}
        )
        # Preserve other columns (take first occurrence)
        other_cols = [c for c in df.columns if c not in [smiles_col, activity_col, "_inchi_key"]]
        if other_cols:
            other_grouped = df.groupby("_inchi_key")[other_cols].first()
            grouped = grouped.join(other_grouped)

        result = grouped.reset_index(drop=True)
    else:
        result = df.drop_duplicates(subset=["_inchi_key"]).drop(columns=["_inchi_key"])
        result = result.reset_index(drop=True)

    if "_inchi_key" in result.columns:
        result = result.drop(columns=["_inchi_key"])

    logger.info(f"De-duplication: {len(df)} → {len(result)} molecules")
    return result


class StandardizationPipeline:
    """
    End-to-end molecule standardization pipeline.

    Applies standardization to a batch of SMILES, removes duplicates,
    and optionally applies quality filters.

    Example::

        pipeline = StandardizationPipeline(
            canonicalize_tautomers=True,
            neutralize_charges=True,
            min_heavy_atoms=3,
            max_heavy_atoms=100,
        )
        clean_df = pipeline.run(raw_df, smiles_col='smiles', activity_col='pIC50')
    """

    def __init__(
        self,
        canonicalize_tautomers: bool = True,
        neutralize_charges: bool = True,
        remove_stereo: bool = False,
        keep_largest_fragment: bool = True,
        min_heavy_atoms: int = 3,
        max_heavy_atoms: int = 150,
        max_molecular_weight: float = 1500.0,
    ):
        self.canonicalize_tautomers = canonicalize_tautomers
        self.neutralize_charges = neutralize_charges
        self.remove_stereo = remove_stereo
        self.keep_largest_fragment = keep_largest_fragment
        self.min_heavy_atoms = min_heavy_atoms
        self.max_heavy_atoms = max_heavy_atoms
        self.max_molecular_weight = max_molecular_weight

    def _passes_filters(self, smiles: str) -> bool:
        """Check if a standardized molecule passes quality filters."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False

        num_heavy = mol.GetNumHeavyAtoms()
        if num_heavy < self.min_heavy_atoms or num_heavy > self.max_heavy_atoms:
            return False

        mw = Descriptors.ExactMolWt(mol)
        if mw > self.max_molecular_weight:
            return False

        return True

    def run(
        self,
        df: pd.DataFrame,
        smiles_col: str = "smiles",
        activity_col: Optional[str] = None,
        aggregation: str = "median",
    ) -> pd.DataFrame:
        """
        Run the full standardization pipeline on a DataFrame.

        Args:
            df: Input DataFrame with SMILES.
            smiles_col: Column name for SMILES strings.
            activity_col: Column name for activity values (optional).
            aggregation: Aggregation method for duplicate activities.

        Returns:
            Cleaned, standardized, de-duplicated DataFrame.
        """
        df = df.copy()
        initial_count = len(df)

        # Step 1: Standardize SMILES
        logger.info(f"Standardizing {initial_count} molecules...")
        df[smiles_col] = df[smiles_col].apply(
            lambda s: standardize_mol(
                s,
                remove_stereo=self.remove_stereo,
                canonicalize_tautomers=self.canonicalize_tautomers,
                neutralize_charges=self.neutralize_charges,
                keep_largest_fragment=self.keep_largest_fragment,
            )
        )

        # Drop failed standardizations
        df = df.dropna(subset=[smiles_col])
        logger.info(f"After standardization: {len(df)} / {initial_count} molecules valid")

        # Step 2: Apply quality filters
        mask = df[smiles_col].apply(self._passes_filters)
        df = df[mask].reset_index(drop=True)
        logger.info(f"After quality filters: {len(df)} molecules")

        # Step 3: Remove duplicates
        df = remove_duplicates(df, smiles_col, activity_col, aggregation)

        logger.info(
            f"Standardization complete: {initial_count} → {len(df)} molecules "
            f"({initial_count - len(df)} removed)"
        )
        return df
