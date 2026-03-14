"""
Counterfactual edit visualization.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    from rdkit import Chem
    from rdkit.Chem import Draw, AllChem
    from rdkit.Chem.Draw import rdMolDraw2D
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


def render_counterfactual_comparison(
    original_smiles: str,
    cf_result: Dict,
    save_path: Optional[str] = None,
    img_size: int = 400,
    figsize: tuple = (14, 5),
) -> Optional[str]:
    """
    Side-by-side comparison of original molecule and counterfactual edit.

    Shows original molecule with removed edges highlighted, and the
    resulting prediction change.

    Args:
        original_smiles: Original SMILES string.
        cf_result: Output dict from CFGNNExplainer.explain().
        save_path: Output path.

    Returns:
        Path to saved image.
    """
    if not HAS_RDKIT:
        raise ImportError("RDKit required for counterfactual visualization")

    mol = Chem.MolFromSmiles(original_smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {original_smiles}")

    AllChem.Compute2DCoords(mol)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Panel 1: Original molecule
    ax = axes[0]
    img = Draw.MolToImage(mol, size=(img_size, img_size))
    ax.imshow(img)
    ax.set_title(f"Original\nPred: {cf_result.get('original_class', '?')}", fontsize=11)
    ax.axis("off")

    # Panel 2: Edge importance mask
    ax = axes[1]
    edge_mask = cf_result.get("edge_mask", None)
    if edge_mask is not None:
        edge_mask_np = edge_mask.numpy() if hasattr(edge_mask, "numpy") else np.array(edge_mask)
        n_bonds = mol.GetNumBonds()
        bond_colors = {}
        bond_list = []
        for i in range(min(n_bonds, len(edge_mask_np) // 2)):
            bond_list.append(i)
            v = edge_mask_np[i * 2] if i * 2 < len(edge_mask_np) else 1.0
            # Red = removed, green = kept
            if v < 0.5:
                bond_colors[i] = (0.9, 0.2, 0.2, 0.8)
            else:
                bond_colors[i] = (0.2, 0.8, 0.2, 0.3)

        drawer = rdMolDraw2D.MolDraw2DSVG(img_size, img_size)
        drawer.DrawMolecule(
            mol,
            highlightBonds=bond_list,
            highlightBondColors=bond_colors,
        )
        drawer.FinishDrawing()

        # Render info text instead
        ax.text(
            0.5, 0.5,
            f"Edges Removed: {cf_result.get('num_edits', 0)}",
            ha="center", va="center", fontsize=14,
            transform=ax.transAxes,
        )
    ax.set_title("Edit Map", fontsize=11)
    ax.axis("off")

    # Panel 3: Prediction change
    ax = axes[2]
    success = cf_result.get("success", False)
    cf_class = cf_result.get("cf_class", "?")
    num_edits = cf_result.get("num_edits", 0)

    info_text = (
        f"CF Class: {cf_class}\n"
        f"Edits: {num_edits}\n"
        f"Success: {'✓' if success else '✗'}"
    )
    bg_color = "#d4edda" if success else "#f8d7da"
    ax.set_facecolor(bg_color)
    ax.text(
        0.5, 0.5, info_text,
        ha="center", va="center", fontsize=14,
        transform=ax.transAxes,
        fontweight="bold",
    )
    ax.set_title("Counterfactual Result", fontsize=11)
    ax.axis("off")

    fig.suptitle("Counterfactual Explanation", fontsize=14, y=1.02)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path

    plt.show()
    plt.close(fig)
    return None


def batch_render_counterfactuals(
    smiles_list: List[str],
    cf_results: List[Dict],
    output_dir: str,
    prefix: str = "cf",
) -> List[str]:
    """Batch-render counterfactual explanations."""
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for i, (smi, result) in enumerate(zip(smiles_list, cf_results)):
        path = os.path.join(output_dir, f"{prefix}_{i:04d}.png")
        try:
            render_counterfactual_comparison(smi, result, save_path=path)
            paths.append(path)
        except Exception:
            continue
    return paths
