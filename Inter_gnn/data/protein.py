"""
Protein graph construction from amino acid sequences.

Provides builders for constructing residue-level graphs from protein
sequences using k-NN graphs in embedding space or contact graphs from
3D structures. Supports both sequence-based and structure-based approaches.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

logger = logging.getLogger(__name__)

# ─── Amino Acid Configuration ─────────────────────────────────────────────────

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: idx for idx, aa in enumerate(AMINO_ACIDS)}

# Physicochemical properties for each amino acid (normalized to [0, 1])
# Properties: hydrophobicity, polarity, molecular_weight, pI, volume
AA_PROPERTIES: Dict[str, List[float]] = {
    "A": [0.62, 0.00, 0.17, 0.57, 0.16],
    "C": [0.29, 0.08, 0.26, 0.47, 0.21],
    "D": [0.00, 1.00, 0.28, 0.16, 0.21],
    "E": [0.00, 1.00, 0.33, 0.19, 0.29],
    "F": [1.00, 0.00, 0.39, 0.49, 0.40],
    "G": [0.48, 0.00, 0.09, 0.55, 0.06],
    "H": [0.14, 0.69, 0.35, 0.73, 0.31],
    "I": [1.00, 0.00, 0.30, 0.55, 0.33],
    "K": [0.00, 1.00, 0.31, 0.95, 0.33],
    "L": [0.97, 0.00, 0.30, 0.55, 0.33],
    "M": [0.74, 0.00, 0.33, 0.50, 0.33],
    "N": [0.00, 0.85, 0.28, 0.47, 0.24],
    "P": [0.72, 0.00, 0.25, 0.58, 0.22],
    "Q": [0.00, 0.85, 0.32, 0.48, 0.29],
    "R": [0.00, 1.00, 0.40, 1.00, 0.37],
    "S": [0.00, 0.23, 0.22, 0.49, 0.16],
    "T": [0.14, 0.23, 0.26, 0.48, 0.22],
    "V": [0.86, 0.00, 0.26, 0.55, 0.27],
    "W": [0.81, 0.08, 0.47, 0.52, 0.47],
    "Y": [0.63, 0.31, 0.42, 0.49, 0.39],
}


def _residue_one_hot(residue: str) -> List[float]:
    """One-hot encode a single amino acid residue."""
    encoding = [0.0] * (len(AMINO_ACIDS) + 1)  # +1 for unknown
    if residue in AA_TO_IDX:
        encoding[AA_TO_IDX[residue]] = 1.0
    else:
        encoding[-1] = 1.0  # unknown residue
    return encoding


def _residue_features(residue: str) -> List[float]:
    """
    Compute feature vector for a single residue.

    Features (~26 dims):
        - One-hot amino acid identity (21 dims)
        - Physicochemical properties (5 dims)
    """
    features = _residue_one_hot(residue)
    props = AA_PROPERTIES.get(residue, [0.0] * 5)
    features.extend(props)
    return features


def _positional_encoding(
    seq_len: int,
    d_model: int = 16,
) -> np.ndarray:
    """
    Sinusoidal positional encoding for residue positions.

    Args:
        seq_len: Length of the protein sequence.
        d_model: Dimension of the positional encoding.

    Returns:
        Array of shape (seq_len, d_model).
    """
    positions = np.arange(seq_len).reshape(-1, 1)
    dim_indices = np.arange(d_model).reshape(1, -1)

    angles = positions / np.power(10000, 2 * (dim_indices // 2) / d_model)
    encoding = np.zeros((seq_len, d_model))
    encoding[:, 0::2] = np.sin(angles[:, 0::2])
    encoding[:, 1::2] = np.cos(angles[:, 1::2])

    return encoding


def _build_knn_edges(
    features: np.ndarray,
    k: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build k-NN graph edges based on feature similarity.

    Uses cosine similarity between residue feature vectors to find
    k nearest neighbors per residue.

    Args:
        features: (N, D) array of residue features.
        k: Number of nearest neighbors.

    Returns:
        Tuple of (edge_index [2, E], edge_weights [E]).
    """
    from scipy.spatial.distance import cdist

    n = features.shape[0]
    k = min(k, n - 1)  # can't have more neighbors than nodes

    if k <= 0:
        return np.zeros((2, 0), dtype=np.int64), np.zeros(0)

    # Cosine distance
    dist_matrix = cdist(features, features, metric="cosine")
    np.fill_diagonal(dist_matrix, np.inf)

    edge_list = []
    edge_weights = []

    for i in range(n):
        # Find k nearest neighbors
        nn_indices = np.argsort(dist_matrix[i])[:k]
        for j in nn_indices:
            edge_list.append([i, j])
            # Convert distance to similarity weight
            edge_weights.append(1.0 - min(dist_matrix[i, j], 1.0))

    edge_index = np.array(edge_list, dtype=np.int64).T
    edge_weights = np.array(edge_weights, dtype=np.float64)

    return edge_index, edge_weights


def _build_sequence_edges(
    seq_len: int,
    window: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build sequential edges based on primary sequence connectivity.

    Connects each residue to its neighbors within a sliding window.

    Args:
        seq_len: Length of the protein sequence.
        window: Size of the connectivity window (each side).

    Returns:
        Tuple of (edge_index [2, E], edge_weights [E]).
    """
    edge_list = []
    edge_weights = []

    for i in range(seq_len):
        for offset in range(-window, window + 1):
            j = i + offset
            if j != i and 0 <= j < seq_len:
                edge_list.append([i, j])
                # Weight decreases with distance
                edge_weights.append(1.0 / (1.0 + abs(offset)))

    if len(edge_list) == 0:
        return np.zeros((2, 0), dtype=np.int64), np.zeros(0)

    edge_index = np.array(edge_list, dtype=np.int64).T
    edge_weights = np.array(edge_weights, dtype=np.float64)

    return edge_index, edge_weights


class ProteinGraphBuilder:
    """
    Builds residue-level graphs from protein sequences.

    Supports two graph construction modes:
      1. **k-NN graph**: Connects residues based on feature similarity in
         embedding space up to k nearest neighbors.
      2. **Contact graph**: Uses 3D structure information to connect residues
         within a distance cutoff (requires coordinates).

    Residue features include one-hot amino acid identity, physicochemical
    properties, and sinusoidal positional encodings.

    Example::

        builder = ProteinGraphBuilder(k=10, pos_encoding_dim=16)
        protein_graph = builder.from_sequence("MKWVTFISLLLLFSSAYS...")
    """

    def __init__(
        self,
        k: int = 10,
        sequence_window: int = 3,
        pos_encoding_dim: int = 16,
        max_seq_len: int = 2000,
        contact_threshold: float = 8.0,
    ):
        """
        Args:
            k: Number of nearest neighbors for k-NN graph construction.
            sequence_window: Window size for sequential edge connectivity.
            pos_encoding_dim: Dimension of sinusoidal positional encoding.
            max_seq_len: Maximum sequence length (longer sequences get truncated).
            contact_threshold: Distance threshold (Å) for contact graph edges.
        """
        self.k = k
        self.sequence_window = sequence_window
        self.pos_encoding_dim = pos_encoding_dim
        self.max_seq_len = max_seq_len
        self.contact_threshold = contact_threshold

    def from_sequence(
        self,
        sequence: str,
        target_id: Optional[str] = None,
    ) -> Optional[Data]:
        """
        Build a protein graph from an amino acid sequence string.

        Constructs a graph using both sequential edges (primary structure)
        and k-NN edges (feature similarity).

        Args:
            sequence: Amino acid sequence (single-letter code).
            target_id: Optional identifier string for the protein.

        Returns:
            PyTorch Geometric Data object representing the protein graph.
        """
        # Clean and truncate sequence
        sequence = sequence.upper().strip()
        sequence = "".join(c for c in sequence if c in AMINO_ACIDS or c == "X")

        if len(sequence) == 0:
            logger.warning("Empty protein sequence provided")
            return None

        if len(sequence) > self.max_seq_len:
            logger.info(
                f"Truncating sequence from {len(sequence)} to {self.max_seq_len} residues"
            )
            sequence = sequence[: self.max_seq_len]

        seq_len = len(sequence)

        # ── Residue features ──
        residue_feats = np.array(
            [_residue_features(aa) for aa in sequence], dtype=np.float32
        )

        # Add positional encoding
        pos_enc = _positional_encoding(seq_len, self.pos_encoding_dim).astype(np.float32)
        node_features = np.concatenate([residue_feats, pos_enc], axis=1)

        # ── Edge construction ──
        # Sequential edges (primary structure)
        seq_edges, seq_weights = _build_sequence_edges(seq_len, self.sequence_window)

        # k-NN edges (feature similarity)
        knn_edges, knn_weights = _build_knn_edges(residue_feats, self.k)

        # Merge edges (remove duplicates)
        all_edges = np.concatenate([seq_edges, knn_edges], axis=1)
        all_weights = np.concatenate([seq_weights, knn_weights])

        # Deduplicate edges (keep highest weight)
        if all_edges.shape[1] > 0:
            edge_tuples = {}
            for idx in range(all_edges.shape[1]):
                key = (all_edges[0, idx], all_edges[1, idx])
                if key not in edge_tuples or all_weights[idx] > edge_tuples[key]:
                    edge_tuples[key] = all_weights[idx]

            deduped_edges = np.array(list(edge_tuples.keys()), dtype=np.int64).T
            deduped_weights = np.array(list(edge_tuples.values()), dtype=np.float32)
        else:
            deduped_edges = np.zeros((2, 0), dtype=np.int64)
            deduped_weights = np.zeros(0, dtype=np.float32)

        # ── Build Data object ──
        data = Data(
            x=torch.tensor(node_features, dtype=torch.float32),
            edge_index=torch.tensor(deduped_edges, dtype=torch.long),
            edge_attr=torch.tensor(deduped_weights, dtype=torch.float32).unsqueeze(-1),
            num_residues=seq_len,
            sequence=sequence,
        )

        if target_id is not None:
            data.target_id = target_id

        return data

    def from_contact_map(
        self,
        sequence: str,
        coordinates: np.ndarray,
        target_id: Optional[str] = None,
    ) -> Optional[Data]:
        """
        Build a protein graph from sequence and 3D C-alpha coordinates.

        Uses distance-based contact graph instead of k-NN similarity.

        Args:
            sequence: Amino acid sequence string.
            coordinates: (N, 3) array of C-alpha atom coordinates.
            target_id: Optional protein identifier.

        Returns:
            Data object with contact-based edges and 3D positions.
        """
        sequence = sequence.upper().strip()
        sequence = "".join(c for c in sequence if c in AMINO_ACIDS or c == "X")

        if len(sequence) == 0:
            return None

        seq_len = min(len(sequence), self.max_seq_len)
        sequence = sequence[:seq_len]
        coordinates = coordinates[:seq_len]

        if coordinates.shape[0] != seq_len:
            logger.warning(
                f"Coordinate count ({coordinates.shape[0]}) != sequence length ({seq_len})"
            )
            min_len = min(coordinates.shape[0], seq_len)
            sequence = sequence[:min_len]
            coordinates = coordinates[:min_len]
            seq_len = min_len

        # ── Residue features ──
        residue_feats = np.array(
            [_residue_features(aa) for aa in sequence], dtype=np.float32
        )
        pos_enc = _positional_encoding(seq_len, self.pos_encoding_dim).astype(np.float32)
        node_features = np.concatenate([residue_feats, pos_enc], axis=1)

        # ── Contact edges ──
        from scipy.spatial.distance import cdist

        dist_matrix = cdist(coordinates, coordinates, metric="euclidean")

        edge_list = []
        edge_weights = []

        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                if dist_matrix[i, j] <= self.contact_threshold:
                    # Bidirectional edges
                    edge_list.append([i, j])
                    edge_list.append([j, i])
                    # Weight inversely proportional to distance
                    w = 1.0 / (1.0 + dist_matrix[i, j])
                    edge_weights.extend([w, w])

        # Also add sequential edges
        seq_edges, seq_weights = _build_sequence_edges(seq_len, self.sequence_window)

        if len(edge_list) > 0:
            contact_edges = np.array(edge_list, dtype=np.int64).T
            contact_weights = np.array(edge_weights, dtype=np.float32)
            all_edges = np.concatenate([contact_edges, seq_edges], axis=1)
            all_weights = np.concatenate([contact_weights, seq_weights])
        else:
            all_edges = seq_edges
            all_weights = seq_weights

        # Deduplicate
        if all_edges.shape[1] > 0:
            edge_tuples = {}
            for idx in range(all_edges.shape[1]):
                key = (all_edges[0, idx], all_edges[1, idx])
                if key not in edge_tuples or all_weights[idx] > edge_tuples[key]:
                    edge_tuples[key] = all_weights[idx]

            final_edges = np.array(list(edge_tuples.keys()), dtype=np.int64).T
            final_weights = np.array(list(edge_tuples.values()), dtype=np.float32)
        else:
            final_edges = np.zeros((2, 0), dtype=np.int64)
            final_weights = np.zeros(0, dtype=np.float32)

        data = Data(
            x=torch.tensor(node_features, dtype=torch.float32),
            edge_index=torch.tensor(final_edges, dtype=torch.long),
            edge_attr=torch.tensor(final_weights, dtype=torch.float32).unsqueeze(-1),
            pos=torch.tensor(coordinates, dtype=torch.float32),
            num_residues=seq_len,
            sequence=sequence,
        )

        if target_id is not None:
            data.target_id = target_id

        return data


# Exported feature dimension constants
RESIDUE_FEATURE_DIM = len(AMINO_ACIDS) + 1 + 5  # one-hot + properties = 26
