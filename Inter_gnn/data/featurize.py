"""
Molecular graph featurization.

Converts SMILES strings to PyTorch Geometric Data objects with rich
atom-level and bond-level features suitable for GNN processing.
Supports optional 3D coordinate generation for distance/angle features.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from torch_geometric.data import Data

# ─── Atom Feature Configuration ───────────────────────────────────────────────

ATOM_TYPES = [
    "C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "Si",
    "B", "Se", "Te", "As", "Ge", "Sn", "Bi", "Sb", "Other",
]

HYBRIDIZATION_TYPES = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]

CHIRALITY_TYPES = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]

FORMAL_CHARGE_RANGE = list(range(-3, 4))  # -3 to +3
NUM_HS_RANGE = list(range(0, 5))  # 0 to 4
DEGREE_RANGE = list(range(0, 7))  # 0 to 6


def _one_hot(value, choices: list) -> List[int]:
    """One-hot encode a value from a fixed set of choices."""
    encoding = [0] * (len(choices) + 1)  # +1 for unknown
    if value in choices:
        encoding[choices.index(value)] = 1
    else:
        encoding[-1] = 1  # unknown category
    return encoding


def atom_features(atom: Chem.Atom) -> List[float]:
    """
    Extract comprehensive atom-level features.

    Features (total ~78 dims):
        - Atom type (one-hot, 20 dims)
        - Degree (one-hot, 8 dims)
        - Formal charge (one-hot, 8 dims)
        - Num Hs (one-hot, 6 dims)
        - Hybridization (one-hot, 6 dims)
        - Chirality type (one-hot, 5 dims)
        - Aromaticity (1 dim)
        - In ring (1 dim)
        - Num radical electrons (1 dim)
        - Atomic mass (scaled, 1 dim)
        - Van der Waals radius (scaled, 1 dim)
        - Covalent radius (scaled, 1 dim)

    Args:
        atom: RDKit Atom object.

    Returns:
        Feature vector as list of floats.
    """
    symbol = atom.GetSymbol()
    if symbol not in ATOM_TYPES[:-1]:
        symbol = "Other"

    features = []

    # Categorical features (one-hot encoded)
    features.extend(_one_hot(symbol, ATOM_TYPES))
    features.extend(_one_hot(atom.GetDegree(), DEGREE_RANGE))
    features.extend(_one_hot(atom.GetFormalCharge(), FORMAL_CHARGE_RANGE))
    features.extend(_one_hot(atom.GetTotalNumHs(), NUM_HS_RANGE))
    features.extend(_one_hot(atom.GetHybridization(), HYBRIDIZATION_TYPES))
    features.extend(_one_hot(atom.GetChiralTag(), CHIRALITY_TYPES))

    # Binary features
    features.append(1.0 if atom.GetIsAromatic() else 0.0)
    features.append(1.0 if atom.IsInRing() else 0.0)
    features.append(float(atom.GetNumRadicalElectrons()))

    # Continuous features (scaled)
    features.append(atom.GetMass() / 100.0)  # scaled atomic mass

    return features


# ─── Bond Feature Configuration ───────────────────────────────────────────────

BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

STEREO_TYPES = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREOCIS,
    Chem.rdchem.BondStereo.STEREOTRANS,
    Chem.rdchem.BondStereo.STEREOANY,
]


def bond_features(bond: Chem.Bond) -> List[float]:
    """
    Extract bond-level features.

    Features (total ~14 dims):
        - Bond type (one-hot, 5 dims)
        - Stereo type (one-hot, 7 dims)
        - Is conjugated (1 dim)
        - Is in ring (1 dim)

    Args:
        bond: RDKit Bond object.

    Returns:
        Feature vector as list of floats.
    """
    features = []
    features.extend(_one_hot(bond.GetBondType(), BOND_TYPES))
    features.extend(_one_hot(bond.GetStereo(), STEREO_TYPES))
    features.append(1.0 if bond.GetIsConjugated() else 0.0)
    features.append(1.0 if bond.IsInRing() else 0.0)
    return features


# ─── Graph Construction ───────────────────────────────────────────────────────


def smiles_to_graph(
    smiles: str,
    y: Optional[torch.Tensor] = None,
    include_hydrogens: bool = False,
) -> Optional[Data]:
    """
    Convert a SMILES string to a PyTorch Geometric Data object.

    Creates a molecular graph where:
        - Nodes = atoms with rich feature vectors
        - Edges = bonds (bidirectional) with feature vectors
        - Optional label `y` for supervised tasks

    Args:
        smiles: Input SMILES string.
        y: Optional target label tensor.
        include_hydrogens: If True, add explicit hydrogens.

    Returns:
        PyTorch Geometric Data object, or None if SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    if include_hydrogens:
        mol = Chem.AddHs(mol)

    # ── Atom features ──
    atom_feat_list = []
    for atom in mol.GetAtoms():
        atom_feat_list.append(atom_features(atom))

    x = torch.tensor(atom_feat_list, dtype=torch.float32)

    # ── Edge features (bidirectional) ──
    edge_indices = []
    edge_attrs = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = bond_features(bond)

        # Add both directions
        edge_indices.append([i, j])
        edge_indices.append([j, i])
        edge_attrs.append(bf)
        edge_attrs.append(bf)

    if len(edge_indices) > 0:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
    else:
        # Single-atom molecule (no bonds)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, len(bond_features(None)) if False else 14), dtype=torch.float32)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        smiles=smiles,
        num_atoms=mol.GetNumAtoms(),
    )

    if y is not None:
        data.y = y

    return data


def smiles_to_3d_graph(
    smiles: str,
    y: Optional[torch.Tensor] = None,
    num_conformers: int = 1,
    max_attempts: int = 200,
    include_hydrogens: bool = False,
) -> Optional[Data]:
    """
    Convert SMILES to a 3D-enriched molecular graph.

    Extends `smiles_to_graph` with:
        - 3D atom coordinates (pos)
        - Pairwise distance features on edges
        - Bond angle features (optional)

    Uses RDKit's ETKDG conformer generator for 3D coordinate assignment.

    Args:
        smiles: Input SMILES string.
        y: Optional target label tensor.
        num_conformers: Number of conformers to generate (best energy selected).
        max_attempts: Max attempts for conformer embedding.
        include_hydrogens: If True, add explicit hydrogens before conformer generation.

    Returns:
        Data object with additional `pos` and distance-augmented `edge_attr`.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)

    # Generate 3D conformers using ETKDG
    params = AllChem.ETKDGv3()
    params.maxAttempts = max_attempts
    params.randomSeed = 42
    params.useSmallRingTorsions = True

    conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_conformers, params=params)
    if len(conf_ids) == 0:
        # Fallback: try without ETKDG constraints
        AllChem.EmbedMolecule(mol, randomSeed=42)
        if mol.GetNumConformers() == 0:
            return smiles_to_graph(smiles, y, include_hydrogens=False)

    # Optimize with MMFF and select lowest energy conformer
    if mol.GetNumConformers() > 1:
        results = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=500)
        energies = [r[1] if r[0] == 0 else float("inf") for r in results]
        best_conf_id = int(np.argmin(energies))
    else:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        best_conf_id = 0

    if not include_hydrogens:
        mol = Chem.RemoveHs(mol)

    # Get base 2D graph
    base_data = smiles_to_graph(Chem.MolToSmiles(mol), y, include_hydrogens)
    if base_data is None:
        return None

    # Re-featurize from the 3D mol to keep atom indexing consistent
    atom_feat_list = [atom_features(atom) for atom in mol.GetAtoms()]
    base_data.x = torch.tensor(atom_feat_list, dtype=torch.float32)

    # Extract 3D coordinates
    conf = mol.GetConformer(best_conf_id)
    positions = []
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        positions.append([pos.x, pos.y, pos.z])

    base_data.pos = torch.tensor(positions, dtype=torch.float32)

    # Augment edge features with 3D distances
    if base_data.edge_index.shape[1] > 0:
        src = base_data.edge_index[0]
        dst = base_data.edge_index[1]
        pos_tensor = base_data.pos

        # Euclidean distances
        diffs = pos_tensor[src] - pos_tensor[dst]
        distances = torch.norm(diffs, dim=1, keepdim=True)

        # Normalized distance (divide by max to bring to [0, 1])
        max_dist = distances.max().clamp(min=1e-8)
        norm_distances = distances / max_dist

        # Gaussian RBF expansion of distances (10 bins, 0–10 Å)
        rbf_centers = torch.linspace(0, 10, 10).unsqueeze(0)
        rbf_gamma = 1.0
        rbf_features = torch.exp(-rbf_gamma * (distances - rbf_centers) ** 2)

        # Concatenate with existing edge features
        base_data.edge_attr = torch.cat(
            [base_data.edge_attr, norm_distances, rbf_features], dim=1
        )

    return base_data


def batch_smiles_to_graphs(
    smiles_list: List[str],
    labels: Optional[List] = None,
    use_3d: bool = False,
    num_workers: int = 0,
) -> List[Data]:
    """
    Convert a batch of SMILES strings to Data objects.

    Args:
        smiles_list: List of SMILES strings.
        labels: Optional list of labels (same length as smiles_list).
        use_3d: If True, generate 3D-enriched graphs.
        num_workers: Number of parallel workers (0 for sequential).

    Returns:
        List of valid Data objects (invalid SMILES are skipped).
    """
    converter = smiles_to_3d_graph if use_3d else smiles_to_graph
    data_list = []

    for idx, smi in enumerate(smiles_list):
        y = None
        if labels is not None:
            label = labels[idx]
            if isinstance(label, (int, float)):
                y = torch.tensor([label], dtype=torch.float32)
            elif isinstance(label, (list, np.ndarray)):
                y = torch.tensor(label, dtype=torch.float32)

        graph = converter(smi, y=y)
        if graph is not None:
            graph.idx = idx
            data_list.append(graph)

    return data_list


# ─── Feature dimension constants (for model configuration) ────────────────────

ATOM_FEATURE_DIM = (
    len(ATOM_TYPES) + 1       # atom type
    + len(DEGREE_RANGE) + 1   # degree
    + len(FORMAL_CHARGE_RANGE) + 1  # formal charge
    + len(NUM_HS_RANGE) + 1   # num Hs
    + len(HYBRIDIZATION_TYPES) + 1  # hybridization
    + len(CHIRALITY_TYPES) + 1      # chirality
    + 1  # aromaticity
    + 1  # in ring
    + 1  # radical electrons
    + 1  # atomic mass (scaled)
)

BOND_FEATURE_DIM = (
    len(BOND_TYPES) + 1     # bond type
    + len(STEREO_TYPES) + 1  # stereo
    + 1  # conjugated
    + 1  # in ring
)
