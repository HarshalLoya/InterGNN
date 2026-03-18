"""
Motif library heatmaps and SMARTS overlay visualizations.
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


def plot_motif_activation_heatmap(
    motif_activations: np.ndarray,
    motif_names: Optional[List[str]] = None,
    molecule_labels: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 8),
    cmap: str = "YlOrRd",
) -> Optional[str]:
    """
    Heatmap of motif activations across molecules.

    Args:
        motif_activations: (N_molecules, K_motifs) activation matrix.
        motif_names: Labels for motif columns.
        molecule_labels: Labels for molecule rows.
        save_path: Output path.
        figsize: Figure size.
        cmap: Colormap.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    im = ax.imshow(motif_activations, cmap=cmap, aspect="auto")

    if motif_names:
        ax.set_xticks(range(len(motif_names)))
        ax.set_xticklabels(motif_names, rotation=45, ha="right", fontsize=8)
    else:
        ax.set_xlabel("Motif Index")

    if molecule_labels:
        ax.set_yticks(range(min(len(molecule_labels), motif_activations.shape[0])))
        ax.set_yticklabels(molecule_labels[:motif_activations.shape[0]], fontsize=7)
    else:
        ax.set_ylabel("Molecule Index")

    ax.set_title("Motif Activation Heatmap", fontsize=14)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Activation")
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path

    plt.show()
    plt.close(fig)
    return None


def render_motif_overlay(
    smiles: str,
    motif_atom_indices: Dict[str, List[int]],
    save_path: Optional[str] = None,
    img_size: int = 400,
) -> Optional[str]:
    """
    Render molecule with motif atoms highlighted in different colors.

    Args:
        smiles: SMILES string.
        motif_atom_indices: Dict mapping motif_name → atom indices.
        save_path: Output path.

    Returns:
        Path to saved image.
    """
    if not HAS_RDKIT:
        raise ImportError("RDKit required")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    AllChem.Compute2DCoords(mol)

    # Assign colors to motifs
    color_palette = [
        (0.9, 0.2, 0.2, 0.5),  # red
        (0.2, 0.7, 0.2, 0.5),  # green
        (0.2, 0.2, 0.9, 0.5),  # blue
        (0.9, 0.6, 0.1, 0.5),  # orange
        (0.6, 0.2, 0.8, 0.5),  # purple
        (0.1, 0.8, 0.8, 0.5),  # cyan
        (0.8, 0.2, 0.6, 0.5),  # magenta
        (0.5, 0.5, 0.2, 0.5),  # olive
    ]

    all_atoms = []
    atom_colors = {}
    for i, (motif_name, indices) in enumerate(motif_atom_indices.items()):
        color = color_palette[i % len(color_palette)]
        for idx in indices:
            if idx < mol.GetNumAtoms():
                all_atoms.append(idx)
                atom_colors[idx] = color

    from rdkit.Chem.Draw import rdMolDraw2D
    drawer = rdMolDraw2D.MolDraw2DSVG(img_size, img_size)
    drawer.DrawMolecule(
        mol,
        highlightAtoms=all_atoms,
        highlightAtomColors=atom_colors,
    )
    drawer.FinishDrawing()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w") as f:
            f.write(drawer.GetDrawingText())
        return save_path

    return None
