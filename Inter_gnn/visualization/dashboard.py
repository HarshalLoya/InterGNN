"""
Interactive batch-export explanation dashboard.

Generates a comprehensive HTML report with all explanations for a batch
of molecules, including atom importance, motif overlays, concept activations,
prototype matching, and counterfactual edits.
"""

from __future__ import annotations

import json
import os
import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ExplanationDashboard:
    """
    Batch explanation report generator.

    Collects explanations from multiple modules and generates a unified
    HTML dashboard for interactive exploration.

    Args:
        output_dir: Directory for saving the dashboard and assets.
        title: Dashboard title.
    """

    def __init__(self, output_dir: str, title: str = "InterGNN Explanation Dashboard"):
        self.output_dir = output_dir
        self.title = title
        self.entries: List[Dict] = []

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "assets"), exist_ok=True)

    def add_entry(
        self,
        smiles: str,
        prediction: float,
        target: Optional[float] = None,
        atom_importance: Optional[np.ndarray] = None,
        motif_atoms: Optional[Dict[str, List[int]]] = None,
        concept_activations: Optional[np.ndarray] = None,
        concept_names: Optional[List[str]] = None,
        prototype_idx: Optional[int] = None,
        prototype_distance: Optional[float] = None,
        cf_result: Optional[Dict] = None,
    ):
        """Add a molecule explanation entry to the dashboard."""
        entry = {
            "idx": len(self.entries),
            "smiles": smiles,
            "prediction": float(prediction),
        }

        if target is not None:
            entry["target"] = float(target)

        if atom_importance is not None:
            entry["atom_importance"] = atom_importance.tolist()

        if motif_atoms is not None:
            entry["motif_atoms"] = motif_atoms

        if concept_activations is not None:
            entry["concept_activations"] = concept_activations.tolist()
            if concept_names:
                entry["concept_names"] = concept_names

        if prototype_idx is not None:
            entry["prototype_idx"] = int(prototype_idx)
            entry["prototype_distance"] = float(prototype_distance or 0.0)

        if cf_result is not None:
            serializable_cf = {}
            for k, v in cf_result.items():
                if hasattr(v, "tolist"):
                    serializable_cf[k] = v.tolist()
                elif hasattr(v, "item"):
                    serializable_cf[k] = v.item()
                else:
                    serializable_cf[k] = v
            entry["counterfactual"] = serializable_cf

        self.entries.append(entry)

    def generate(self) -> str:
        """
        Generate the HTML dashboard.

        Returns:
            Path to the generated HTML file.
        """
        # Save data as JSON
        data_path = os.path.join(self.output_dir, "assets", "data.json")
        with open(data_path, "w") as f:
            json.dump(self.entries, f, indent=2)

        # Generate HTML
        html = self._build_html()
        html_path = os.path.join(self.output_dir, "dashboard.html")
        with open(html_path, "w") as f:
            f.write(html)

        logger.info(f"Dashboard generated: {html_path} ({len(self.entries)} entries)")
        return html_path

    def _build_html(self) -> str:
        """Build the HTML dashboard."""
        rows_html = []
        for entry in self.entries:
            imp_str = ""
            if "atom_importance" in entry:
                imp = np.array(entry["atom_importance"])
                imp_str = f"Max: {imp.max():.3f}, Mean: {imp.mean():.3f}"

            proto_str = ""
            if "prototype_idx" in entry:
                proto_str = f"Proto {entry['prototype_idx']} (d={entry['prototype_distance']:.3f})"

            cf_str = ""
            if "counterfactual" in entry:
                cf = entry["counterfactual"]
                cf_str = f"{'✓' if cf.get('success') else '✗'} ({cf.get('num_edits', 0)} edits)"

            concept_str = ""
            if "concept_activations" in entry:
                acts = np.array(entry["concept_activations"])
                top_idx = np.argsort(np.abs(acts))[-3:][::-1]
                names = entry.get("concept_names", [f"C{i}" for i in range(len(acts))])
                concept_str = ", ".join([f"{names[i]}={acts[i]:.2f}" for i in top_idx])

            row = f"""
            <tr>
                <td>{entry['idx']}</td>
                <td><code>{entry['smiles'][:40]}</code></td>
                <td>{entry['prediction']:.4f}</td>
                <td>{entry.get('target', 'N/A')}</td>
                <td>{imp_str}</td>
                <td>{proto_str}</td>
                <td>{concept_str}</td>
                <td>{cf_str}</td>
            </tr>"""
            rows_html.append(row)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{self.title}</title>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; margin: 20px; background: #f5f6fa; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        .summary {{ background: #fff; padding: 20px; border-radius: 8px;
                     box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }}
        table {{ border-collapse: collapse; width: 100%; background: #fff;
                 border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        th {{ background: #3498db; color: #fff; padding: 12px 8px; text-align: left; font-size: 13px; }}
        td {{ padding: 10px 8px; border-bottom: 1px solid #ecf0f1; font-size: 12px; }}
        tr:hover {{ background: #f1f8ff; }}
        code {{ background: #ecf0f1; padding: 2px 6px; border-radius: 3px; font-size: 11px; }}
        .stat {{ display: inline-block; margin: 5px 15px; }}
        .stat-val {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
        .stat-label {{ font-size: 12px; color: #7f8c8d; }}
    </style>
</head>
<body>
    <h1>{self.title}</h1>
    <div class="summary">
        <div class="stat">
            <div class="stat-val">{len(self.entries)}</div>
            <div class="stat-label">Molecules</div>
        </div>
        <div class="stat">
            <div class="stat-val">{sum(1 for e in self.entries if 'atom_importance' in e)}</div>
            <div class="stat-label">With Importance</div>
        </div>
        <div class="stat">
            <div class="stat-val">{sum(1 for e in self.entries if 'prototype_idx' in e)}</div>
            <div class="stat-label">With Prototypes</div>
        </div>
        <div class="stat">
            <div class="stat-val">{sum(1 for e in self.entries if 'counterfactual' in e)}</div>
            <div class="stat-label">With Counterfactuals</div>
        </div>
    </div>
    <table>
        <thead>
            <tr>
                <th>#</th><th>SMILES</th><th>Prediction</th><th>Target</th>
                <th>Importance</th><th>Prototype</th><th>Top Concepts</th><th>Counterfactual</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows_html)}
        </tbody>
    </table>
</body>
</html>"""
