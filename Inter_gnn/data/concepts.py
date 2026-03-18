"""
Domain concept library and matcher.

Provides a curated library of SMARTS-based chemical concepts (pharmacophores,
functional groups, PAINS alerts) for concept whitening alignment, and tools
to match these concepts against molecules to produce binary concept vectors.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from rdkit import Chem

logger = logging.getLogger(__name__)

# ─── SMARTS Concept Library ──────────────────────────────────────────────────

SMARTS_LIBRARY: Dict[str, Dict[str, str]] = {
    # ── Hydrogen Bond Donors & Acceptors ──
    "hbd_oh": {
        "smarts": "[OX2H]",
        "description": "Hydroxyl group (hydrogen bond donor)",
        "category": "pharmacophore",
    },
    "hbd_nh": {
        "smarts": "[NX3;H2,H1;!$(NC=O)]",
        "description": "Primary/secondary amine (hydrogen bond donor)",
        "category": "pharmacophore",
    },
    "hbd_nh_amide": {
        "smarts": "[NX3H]C(=O)",
        "description": "Amide NH (hydrogen bond donor)",
        "category": "pharmacophore",
    },
    "hba_carbonyl": {
        "smarts": "[CX3]=[OX1]",
        "description": "Carbonyl oxygen (hydrogen bond acceptor)",
        "category": "pharmacophore",
    },
    "hba_ether": {
        "smarts": "[OX2H0]",
        "description": "Ether oxygen (hydrogen bond acceptor)",
        "category": "pharmacophore",
    },
    "hba_nitrogen": {
        "smarts": "[nX2]",
        "description": "Aromatic nitrogen (hydrogen bond acceptor)",
        "category": "pharmacophore",
    },
    # ── Aromatic Systems ──
    "aromatic_ring_6": {
        "smarts": "c1ccccc1",
        "description": "Benzene ring",
        "category": "scaffold",
    },
    "aromatic_ring_5": {
        "smarts": "[#6]1~[#6]~[#6]~[#6]~[#6]1",
        "description": "Five-membered aromatic ring",
        "category": "scaffold",
    },
    "heterocyclic_n": {
        "smarts": "[nR1]",
        "description": "Nitrogen in aromatic heterocycle",
        "category": "scaffold",
    },
    "heterocyclic_o": {
        "smarts": "[oR1]",
        "description": "Oxygen in aromatic heterocycle",
        "category": "scaffold",
    },
    "heterocyclic_s": {
        "smarts": "[sR1]",
        "description": "Sulfur in aromatic heterocycle",
        "category": "scaffold",
    },
    # ── Functional Groups ──
    "carboxylic_acid": {
        "smarts": "[CX3](=O)[OX2H1]",
        "description": "Carboxylic acid",
        "category": "functional_group",
    },
    "ester": {
        "smarts": "[CX3](=O)[OX2H0]",
        "description": "Ester group",
        "category": "functional_group",
    },
    "amide": {
        "smarts": "[NX3][CX3](=[OX1])[#6]",
        "description": "Amide bond",
        "category": "functional_group",
    },
    "sulfonamide": {
        "smarts": "[#16X4](=[OX1])(=[OX1])([NX3])",
        "description": "Sulfonamide group",
        "category": "functional_group",
    },
    "nitro": {
        "smarts": "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
        "description": "Nitro group",
        "category": "functional_group",
    },
    "halide_f": {
        "smarts": "[F]",
        "description": "Fluorine substituent",
        "category": "functional_group",
    },
    "halide_cl": {
        "smarts": "[Cl]",
        "description": "Chlorine substituent",
        "category": "functional_group",
    },
    "halide_br": {
        "smarts": "[Br]",
        "description": "Bromine substituent",
        "category": "functional_group",
    },
    "phosphate": {
        "smarts": "[PX4](=O)([OX2])[OX2]",
        "description": "Phosphate group",
        "category": "functional_group",
    },
    # ── PAINS Alerts (Pan-Assay Interference Compounds) ──
    "pains_quinone": {
        "smarts": "[#6]1(=[OX1])[#6]~[#6][#6](=[OX1])[#6]~[#6]1",
        "description": "Quinone (PAINS alert)",
        "category": "pains",
    },
    "pains_catechol": {
        "smarts": "c1cc(O)c(O)cc1",
        "description": "Catechol (PAINS alert)",
        "category": "pains",
    },
    "pains_michael_acceptor": {
        "smarts": "[#6]=[#6]-[CX3]=[OX1]",
        "description": "Michael acceptor enone (PAINS alert)",
        "category": "pains",
    },
    "pains_rhodanine": {
        "smarts": "O=C1CSC(=S)N1",
        "description": "Rhodanine (PAINS alert)",
        "category": "pains",
    },
    "pains_hydroxyphenyl_hydrazone": {
        "smarts": "c1ccc(O)cc1/N=N/c",
        "description": "Hydroxyphenyl hydrazone (PAINS alert)",
        "category": "pains",
    },
    # ── Pharmacophore Features ──
    "hydrophobic_chain": {
        "smarts": "[CX4][CX4][CX4][CX4]",
        "description": "Aliphatic chain (hydrophobic)",
        "category": "pharmacophore",
    },
    "positive_charge_center": {
        "smarts": "[+1,+2,+3]",
        "description": "Positively charged center",
        "category": "pharmacophore",
    },
    "negative_charge_center": {
        "smarts": "[-1,-2,-3]",
        "description": "Negatively charged center",
        "category": "pharmacophore",
    },
    # ── Drug-like Features ──
    "rotatable_bond": {
        "smarts": "[!$([NH]!@C(=O))&!D1]-&!@[!$([NH]!@C(=O))&!D1]",
        "description": "Rotatable bond (flexibility indicator)",
        "category": "drug_likeness",
    },
    "chiral_center": {
        "smarts": "[C@@H]",
        "description": "Chiral carbon center",
        "category": "drug_likeness",
    },
}


def _compile_smarts(smarts_str: str) -> Optional[Chem.Mol]:
    """Compile a SMARTS pattern and return None on failure."""
    pattern = Chem.MolFromSmarts(smarts_str)
    if pattern is None:
        logger.warning(f"Failed to compile SMARTS: {smarts_str}")
    return pattern


# Pre-compile SMARTS patterns for performance
_COMPILED_PATTERNS: Dict[str, Optional[Chem.Mol]] = {}


def _get_compiled_patterns() -> Dict[str, Chem.Mol]:
    """Lazy-compile and cache SMARTS patterns."""
    global _COMPILED_PATTERNS
    if not _COMPILED_PATTERNS:
        for name, info in SMARTS_LIBRARY.items():
            pattern = _compile_smarts(info["smarts"])
            if pattern is not None:
                _COMPILED_PATTERNS[name] = pattern
    return _COMPILED_PATTERNS


def match_concepts(
    mol: Chem.Mol,
    library: Optional[Dict[str, Dict[str, str]]] = None,
    return_match_positions: bool = False,
) -> Dict[str, any]:
    """
    Match a molecule against the SMARTS concept library.

    Args:
        mol: RDKit Mol object.
        library: Optional custom library (defaults to SMARTS_LIBRARY).
        return_match_positions: If True, include atom indices of each match.

    Returns:
        Dictionary with:
            - 'vector': binary concept vector (1 if concept present, 0 otherwise)
            - 'concept_names': ordered list of concept names
            - 'matches': dict of concept_name → list of match atom tuples (if requested)
            - 'counts': dict of concept_name → number of occurrences
    """
    if library is None:
        library = SMARTS_LIBRARY

    patterns = _get_compiled_patterns()
    concept_names = list(patterns.keys())
    vector = np.zeros(len(concept_names), dtype=np.float32)
    matches = {}
    counts = {}

    for idx, name in enumerate(concept_names):
        if name not in library:
            continue
        pattern = patterns[name]
        mol_matches = mol.GetSubstructMatches(pattern)
        count = len(mol_matches)
        counts[name] = count

        if count > 0:
            vector[idx] = 1.0
            if return_match_positions:
                matches[name] = [list(m) for m in mol_matches]

    result = {
        "vector": vector,
        "concept_names": concept_names,
        "counts": counts,
    }
    if return_match_positions:
        result["matches"] = matches

    return result


def batch_match_concepts(
    smiles_list: List[str],
    library: Optional[Dict[str, Dict[str, str]]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute concept vectors for a batch of molecules.

    Args:
        smiles_list: List of SMILES strings.
        library: Optional custom concept library.

    Returns:
        Tuple of:
            - concept_matrix: (N, num_concepts) binary array
            - concept_names: list of concept names (column labels)
    """
    results = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            result = match_concepts(mol, library)
            results.append(result["vector"])
        else:
            # Invalid molecule: zero vector
            num_concepts = len(_get_compiled_patterns())
            results.append(np.zeros(num_concepts, dtype=np.float32))

    concept_matrix = np.stack(results, axis=0)
    concept_names = list(_get_compiled_patterns().keys())
    return concept_matrix, concept_names


class ConceptDataset:
    """
    Holds concept examples for concept-whitening alignment training.

    For each concept axis, stores positive and negative example molecule
    indices from the training set. These are used during training to align
    the latent space axes with predefined chemical concepts.

    Example::

        concept_ds = ConceptDataset(smiles_list, labels)
        concept_ds.build_concept_examples(min_examples=5)
        pos_indices, neg_indices = concept_ds.get_examples("hbd_oh")
    """

    def __init__(
        self,
        smiles_list: List[str],
        library: Optional[Dict[str, Dict[str, str]]] = None,
    ):
        self.smiles_list = smiles_list
        self.library = library or SMARTS_LIBRARY
        self.concept_matrix, self.concept_names = batch_match_concepts(
            smiles_list, self.library
        )
        self._positive_indices: Dict[str, np.ndarray] = {}
        self._negative_indices: Dict[str, np.ndarray] = {}

    def build_concept_examples(self, min_examples: int = 5) -> List[str]:
        """
        Build positive/negative example sets for each concept.

        Args:
            min_examples: Minimum number of positive examples required
                to keep a concept active.

        Returns:
            List of concept names that have enough examples.
        """
        active_concepts = []

        for idx, name in enumerate(self.concept_names):
            pos = np.where(self.concept_matrix[:, idx] > 0)[0]
            neg = np.where(self.concept_matrix[:, idx] == 0)[0]

            if len(pos) >= min_examples:
                self._positive_indices[name] = pos
                self._negative_indices[name] = neg
                active_concepts.append(name)

        logger.info(
            f"Built concept examples: {len(active_concepts)}/{len(self.concept_names)} "
            f"concepts have >= {min_examples} positive examples"
        )
        return active_concepts

    def get_examples(
        self, concept_name: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get positive and negative example indices for a concept."""
        return (
            self._positive_indices.get(concept_name, np.array([])),
            self._negative_indices.get(concept_name, np.array([])),
        )

    def get_concept_vector(self, idx: int) -> torch.Tensor:
        """Get the concept vector for a molecule by index."""
        return torch.tensor(self.concept_matrix[idx], dtype=torch.float32)

    @property
    def num_concepts(self) -> int:
        return len(self.concept_names)

    @property
    def num_active_concepts(self) -> int:
        return len(self._positive_indices)


# Exported constant
NUM_CONCEPTS = len(SMARTS_LIBRARY)
