"""
Concept activation bar chart visualization.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt


def plot_concept_activations(
    activations: np.ndarray,
    concept_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6),
    top_k: int = 15,
    title: str = "Concept Activations",
) -> Optional[str]:
    """
    Bar chart of concept activations for a single sample.

    Args:
        activations: (num_concepts,) activation values.
        concept_names: Human-readable concept names.
        save_path: Output path.
        top_k: Show only top-k most activated concepts.
        title: Plot title.

    Returns:
        Path to saved image.
    """
    n = len(activations)
    if concept_names is None:
        concept_names = [f"Concept {i}" for i in range(n)]

    # Sort by activation
    sorted_idx = np.argsort(np.abs(activations))[::-1][:top_k]
    sorted_names = [concept_names[i] for i in sorted_idx]
    sorted_vals = activations[sorted_idx]

    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in sorted_vals]

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    bars = ax.barh(range(len(sorted_vals)), sorted_vals, color=colors, edgecolor="white")
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=9)
    ax.set_xlabel("Activation Value")
    ax.set_title(title, fontsize=14)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax.invert_yaxis()
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path

    plt.show()
    plt.close(fig)
    return None


def plot_concept_comparison(
    activations_list: List[np.ndarray],
    sample_labels: List[str],
    concept_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (14, 8),
    top_k: int = 10,
) -> Optional[str]:
    """
    Compare concept activations across multiple samples.

    Args:
        activations_list: List of activation arrays.
        sample_labels: Label for each sample.
        concept_names: Concept names.
        save_path: Output path.
        top_k: Number of concepts to show.

    Returns:
        Path to saved image.
    """
    n_concepts = len(activations_list[0])
    if concept_names is None:
        concept_names = [f"C{i}" for i in range(n_concepts)]

    # Find most variable concepts
    all_acts = np.array(activations_list)
    variance = np.var(all_acts, axis=0)
    top_idx = np.argsort(variance)[::-1][:top_k]

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    x = np.arange(len(top_idx))
    width = 0.8 / len(activations_list)

    for i, (acts, label) in enumerate(zip(activations_list, sample_labels)):
        offset = (i - len(activations_list) / 2 + 0.5) * width
        ax.bar(x + offset, acts[top_idx], width, label=label, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([concept_names[i] for i in top_idx], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Activation")
    ax.set_title("Concept Activation Comparison", fontsize=14)
    ax.legend(fontsize=9)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path

    plt.show()
    plt.close(fig)
    return None
