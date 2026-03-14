"""
Prototype gallery visualization.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt

try:
    from rdkit import Chem
    from rdkit.Chem import Draw, AllChem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


def plot_prototype_gallery(
    prototype_embeddings: np.ndarray,
    nearest_smiles: Dict[int, List[str]],
    save_path: Optional[str] = None,
    num_examples_per_proto: int = 3,
    img_size: int = 250,
    figsize: Optional[tuple] = None,
) -> Optional[str]:
    """
    Render a gallery of prototypes with their nearest training examples.

    Args:
        prototype_embeddings: (num_protos, D) prototype vectors.
        nearest_smiles: Dict mapping prototype index → list of nearest SMILES.
        save_path: Path to save the gallery image.
        num_examples_per_proto: Number of examples per prototype.
        img_size: Size of each molecule image.
        figsize: Optional figure size.

    Returns:
        Path to saved image.
    """
    if not HAS_RDKIT:
        raise ImportError("RDKit required for prototype visualization")

    num_protos = len(nearest_smiles)
    cols = num_examples_per_proto
    rows = num_protos

    if figsize is None:
        figsize = (cols * 3, rows * 3)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1:
        axes = axes.reshape(1, -1)
    if cols == 1:
        axes = axes.reshape(-1, 1)

    for proto_idx in range(num_protos):
        smiles_list = nearest_smiles.get(proto_idx, [])
        for j in range(cols):
            ax = axes[proto_idx, j]
            ax.axis("off")

            if j < len(smiles_list):
                mol = Chem.MolFromSmiles(smiles_list[j])
                if mol is not None:
                    img = Draw.MolToImage(mol, size=(img_size, img_size))
                    ax.imshow(img)

            if j == 0:
                ax.set_ylabel(f"Proto {proto_idx}", fontsize=10, rotation=0, labelpad=40)

    fig.suptitle("Prototype Gallery: Nearest Training Examples", fontsize=14, y=1.02)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path

    plt.show()
    plt.close(fig)
    return None


def plot_prototype_distances(
    distances: np.ndarray,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
) -> Optional[str]:
    """
    Plot distribution of distances from samples to their nearest prototype.

    Args:
        distances: (N, num_protos) distance matrix.
        save_path: Path to save plot.

    Returns:
        Path to saved image.
    """
    # Robustly filter out NaN / Inf rows before plotting
    finite_mask = np.isfinite(distances).all(axis=1)
    distances = distances[finite_mask]
    if distances.size == 0:
        # Nothing plottable – save an empty figure with a note
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No finite prototype distances\n(model may need more training)",
                ha="center", va="center", fontsize=12)
        ax.axis("off")
        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return save_path
        plt.show(); plt.close(fig)
        return None

    min_dists = distances.min(axis=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Histogram of min distances
    ax1.hist(min_dists, bins=50, color="#3498db", edgecolor="white", alpha=0.8)
    ax1.set_xlabel("Distance to Nearest Prototype")
    ax1.set_ylabel("Count")
    ax1.set_title("Distance Distribution")

    # Heatmap of prototype usage
    nearest_proto = distances.argmin(axis=1)
    proto_counts = np.bincount(nearest_proto, minlength=distances.shape[1])
    ax2.bar(range(len(proto_counts)), proto_counts, color="#2ecc71", edgecolor="white")
    ax2.set_xlabel("Prototype Index")
    ax2.set_ylabel("Assignments")
    ax2.set_title("Prototype Utilization")

    fig.suptitle("Prototype Analysis", fontsize=14)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path

    plt.show()
    plt.close(fig)
    return None
