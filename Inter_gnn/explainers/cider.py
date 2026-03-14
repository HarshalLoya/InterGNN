"""
CIDER-style causal invariance diagnostics.

Tests whether learned explanations capture causally invariant features
by checking explanation consistency across different data environments
(e.g., scaffold groups, assay contexts).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


class CIDERDiagnostics:
    """
    CIDER-inspired Causal Invariance Diagnostics.

    Tests whether model explanations identify causally invariant features
    versus spurious correlations by evaluating explanation consistency
    across different data environments/domains.

    Key diagnostic tests:
      1. Cross-environment consistency: same molecule, different contexts
      2. Invariance score: how stable are top-k features across environments
      3. Spurious feature detection: features that only predict in one env.

    Args:
        model: Trained InterGNN model.
        k: Number of top features to consider for invariance.
    """

    def __init__(self, model: nn.Module, k: int = 10):
        self.model = model
        self.k = k

    @torch.no_grad()
    def _get_node_importance(self, data: Data) -> torch.Tensor:
        """Extract node importance via gradient norm."""
        self.model.eval()
        device = data.x.device
        x = data.x.clone().requires_grad_(True)
        batch = torch.zeros(x.shape[0], dtype=torch.long, device=device)

        out = self.model.mol_encoder(x, data.edge_index, data.edge_attr, batch)
        z = out["graph_embedding"]
        pred = self.model.task_head(z)

        grad = torch.autograd.grad(
            pred.sum(), x, create_graph=False, retain_graph=False
        )[0]
        importance = torch.norm(grad, dim=-1)
        return importance

    def compute_invariance_score(
        self,
        data_environments: Dict[str, List[Data]],
    ) -> Dict:
        """
        Compute explanation invariance across environments.

        For each molecule appearing in multiple environments, measures
        how consistent the top-k important atoms are.

        Args:
            data_environments: Dict mapping environment name to list of Data objects.
                Each Data object should have a 'smiles' attribute for matching.

        Returns:
            Dict with invariance scores and per-environment statistics.
        """
        # Group data by molecule (SMILES) across environments
        mol_env_importances: Dict[str, Dict[str, torch.Tensor]] = {}

        for env_name, data_list in data_environments.items():
            for data in data_list:
                smi = getattr(data, "smiles", None)
                if smi is None:
                    continue

                importance = self._get_node_importance(data)
                if smi not in mol_env_importances:
                    mol_env_importances[smi] = {}
                mol_env_importances[smi][env_name] = importance

        # Compute invariance for molecules in 2+ environments
        invariance_scores = []
        per_mol_results = []

        for smi, env_imps in mol_env_importances.items():
            if len(env_imps) < 2:
                continue

            env_names = list(env_imps.keys())
            topk_sets = {}

            for env_name in env_names:
                imp = env_imps[env_name]
                k = min(self.k, len(imp))
                topk = imp.topk(k).indices.cpu().numpy()
                topk_sets[env_name] = set(topk.tolist())

            # Pairwise Jaccard similarity of top-k sets
            jaccard_scores = []
            for i in range(len(env_names)):
                for j in range(i + 1, len(env_names)):
                    set_i = topk_sets[env_names[i]]
                    set_j = topk_sets[env_names[j]]
                    if len(set_i | set_j) == 0:
                        continue
                    jaccard = len(set_i & set_j) / len(set_i | set_j)
                    jaccard_scores.append(jaccard)

            if jaccard_scores:
                mean_jaccard = float(np.mean(jaccard_scores))
                invariance_scores.append(mean_jaccard)
                per_mol_results.append({
                    "smiles": smi,
                    "num_environments": len(env_imps),
                    "invariance_score": mean_jaccard,
                    "top_k_overlap": jaccard_scores,
                })

        overall_invariance = float(np.mean(invariance_scores)) if invariance_scores else 0.0

        return {
            "overall_invariance": overall_invariance,
            "num_molecules_tested": len(invariance_scores),
            "invariance_std": float(np.std(invariance_scores)) if invariance_scores else 0.0,
            "per_molecule": per_mol_results,
        }

    def detect_spurious_features(
        self,
        data_environments: Dict[str, List[Data]],
        consistency_threshold: float = 0.3,
    ) -> Dict:
        """
        Detect potentially spurious features.

        A feature is flagged as potentially spurious if it appears important
        in one environment but not others (low cross-environment consistency).

        Args:
            data_environments: Environment-grouped data.
            consistency_threshold: Below this Jaccard score → spurious flag.

        Returns:
            Dict with suspected spurious features and diagnostics.
        """
        invariance_result = self.compute_invariance_score(data_environments)

        spurious_molecules = []
        for mol_result in invariance_result.get("per_molecule", []):
            if mol_result["invariance_score"] < consistency_threshold:
                spurious_molecules.append(mol_result)

        return {
            "num_spurious_flagged": len(spurious_molecules),
            "total_tested": invariance_result["num_molecules_tested"],
            "fraction_spurious": (
                len(spurious_molecules) / max(invariance_result["num_molecules_tested"], 1)
            ),
            "spurious_molecules": spurious_molecules,
            "consistency_threshold": consistency_threshold,
        }

    def run_full_diagnostics(
        self,
        data_environments: Dict[str, List[Data]],
    ) -> Dict:
        """
        Run all CIDER diagnostic tests.

        Returns comprehensive report including invariance scores,
        spurious feature detection, and per-environment statistics.
        """
        logger.info(f"Running CIDER diagnostics across {len(data_environments)} environments")

        invariance = self.compute_invariance_score(data_environments)
        spurious = self.detect_spurious_features(data_environments)

        # Per-environment summary
        env_stats = {}
        for env_name, data_list in data_environments.items():
            importances = []
            for data in data_list:
                imp = self._get_node_importance(data)
                importances.append(imp.mean().item())
            env_stats[env_name] = {
                "num_samples": len(data_list),
                "mean_importance": float(np.mean(importances)) if importances else 0.0,
                "std_importance": float(np.std(importances)) if importances else 0.0,
            }

        return {
            "invariance": invariance,
            "spurious_detection": spurious,
            "environment_stats": env_stats,
        }
