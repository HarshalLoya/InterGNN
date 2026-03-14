"""
Molecular explanation visualization: atom/bond saliency rendering.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

try:
    from rdkit import Chem
    from rdkit.Chem import Draw, AllChem
    from rdkit.Chem.Draw import rdMolDraw2D
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


def _importance_to_colors(
    importance: np.ndarray,
    cmap_name: str = "RdYlGn_r",
) -> List[Tuple[float, ...]]:
    """Convert importance scores to RGBA colors."""
    norm = Normalize(vmin=importance.min(), vmax=importance.max() + 1e-8)
    cmap = plt.get_cmap(cmap_name)
    return [cmap(norm(v)) for v in importance]


def render_atom_importance(
    smiles: str,
    atom_importance: np.ndarray,
    save_path: Optional[str] = None,
    img_size: Tuple[int, int] = (600, 400),
    cmap: str = "RdYlGn_r",
    title: Optional[str] = None,
) -> Optional[str]:
    """
    Render molecule with atoms colored by importance.

    Args:
        smiles: Input SMILES string.
        atom_importance: (N_atoms,) importance scores.
        save_path: Path to save image (PNG/SVG). Returns path if saved.
        img_size: Image dimensions (width, height).
        cmap: Matplotlib colormap name.
        title: Optional title for the image.

    Returns:
        Path to saved image, or None.
    """
    if not HAS_RDKIT:
        raise ImportError("RDKit required for molecular visualization")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    AllChem.Compute2DCoords(mol)

    n_atoms = mol.GetNumAtoms()
    if len(atom_importance) != n_atoms:
        # Truncate or pad
        imp = np.zeros(n_atoms)
        imp[:min(len(atom_importance), n_atoms)] = atom_importance[:n_atoms]
    else:
        imp = atom_importance

    # Generate atom colors — cast to native Python float for RDKit C++ interop
    colors = _importance_to_colors(imp, cmap)
    atom_colors = {i: tuple(float(c) for c in colors[i]) for i in range(n_atoms)}
    radii = {i: float(0.3 + 0.2 * (imp[i] / (imp.max() + 1e-8))) for i in range(n_atoms)}

    # Draw using RDKit
    drawer = rdMolDraw2D.MolDraw2DSVG(img_size[0], img_size[1])
    opts = drawer.drawOptions()
    opts.clearBackground = True

    drawer.DrawMolecule(
        mol,
        highlightAtoms=list(range(n_atoms)),
        highlightAtomColors=atom_colors,
        highlightAtomRadii=radii,
    )
    drawer.FinishDrawing()
    svg_text = drawer.GetDrawingText()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        if save_path.endswith(".svg"):
            with open(save_path, "w") as f:
                f.write(svg_text)
        else:
            # Save as PNG using matplotlib
            fig, ax = plt.subplots(1, 1, figsize=(img_size[0] / 100, img_size[1] / 100))
            img = Draw.MolToImage(mol, size=img_size, kekulize=True)
            ax.imshow(img)
            ax.axis("off")
            if title:
                ax.set_title(title, fontsize=10)
            fig.tight_layout()
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        return save_path

    return None


def render_bond_importance(
    smiles: str,
    bond_importance: np.ndarray,
    save_path: Optional[str] = None,
    img_size: Tuple[int, int] = (600, 400),
    cmap: str = "YlOrRd",
) -> Optional[str]:
    """
    Render molecule with bonds colored by importance.

    Args:
        smiles: SMILES string.
        bond_importance: (N_bonds,) importance scores.
        save_path: Path to save image.

    Returns:
        Path to saved image, or None.
    """
    if not HAS_RDKIT:
        raise ImportError("RDKit required for molecular visualization")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    AllChem.Compute2DCoords(mol)

    n_bonds = mol.GetNumBonds()
    imp = np.zeros(n_bonds)
    imp[:min(len(bond_importance), n_bonds)] = bond_importance[:n_bonds]

    colors = _importance_to_colors(imp, cmap)
    bond_colors = {i: colors[i] for i in range(n_bonds)}

    drawer = rdMolDraw2D.MolDraw2DSVG(img_size[0], img_size[1])
    drawer.DrawMolecule(
        mol,
        highlightBonds=list(range(n_bonds)),
        highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w") as f:
            f.write(drawer.GetDrawingText())
        return save_path

    return None


def batch_render_explanations(
    smiles_list: List[str],
    importance_list: List[np.ndarray],
    output_dir: str,
    prefix: str = "explanation",
    cmap: str = "RdYlGn_r",
) -> List[str]:
    """
    Batch-render atom importance visualizations.

    Returns list of saved file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for i, (smi, imp) in enumerate(zip(smiles_list, importance_list)):
        path = os.path.join(output_dir, f"{prefix}_{i:04d}.svg")
        try:
            render_atom_importance(smi, imp, save_path=path, cmap=cmap)
            paths.append(path)
        except Exception:
            continue
    return paths
