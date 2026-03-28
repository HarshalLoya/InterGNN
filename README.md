# InterGNN — Interpretable GNN-Based Framework for Drug Discovery

<p align="center">
  <strong>An end-to-end interpretable Graph Neural Network framework combining state-of-the-art molecular property prediction with intrinsic and post-hoc explainability methods for drug discovery and candidate screening.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9%2B-blue" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-red" alt="PyTorch">
  <img src="https://img.shields.io/badge/PyG-2.4%2B-orange" alt="PyG">
</p>

*This repository contains the codebase for a B.Tech Final Year Project focusing on interpretability in Graph Neural Networks for drug-target interaction and molecular property prediction.*

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Configuration](#1-create-a-configuration-file)
  - [Training](#2-training)
  - [Evaluation](#3-evaluation)
  - [Explanation](#4-generating-explanations)
  - [Dashboard](#5-explanation-dashboard)
- [Python API](#python-api)
  - [Data Pipeline](#data-pipeline)
  - [Model Construction](#model-construction)
  - [Training Pipeline](#training-pipeline)
  - [Explanations](#generating-explanations)
  - [Evaluation](#evaluation-api)
  - [Visualization](#visualization-api)
- [Module Reference](#module-reference)
  - [Data & Preprocessing](#data--preprocessing-inter_gnndata)
  - [Core Model](#core-model-inter_gnnmodels)
  - [Intrinsic Interpretability](#intrinsic-interpretability-inter_gnninterpretability)
  - [Post-hoc Explainers](#post-hoc-explainers-inter_gnnexplainers)
  - [Training Pipeline](#training-pipeline-inter_gnntraining)
  - [Evaluation Metrics](#evaluation-metrics-inter_gnnevaluation)
  - [Visualization](#visualization-inter_gnnvisualization)
- [Supported Datasets](#supported-datasets)
- [Two-Phase Training Strategy](#two-phase-training-strategy)
- [Interpretability Deep Dive](#interpretability-deep-dive)
  - [PAGE Prototypes](#page-prototypes)
  - [MAGE Motifs](#mage-motifs)
  - [Concept Whitening](#concept-whitening)
  - [Activity Cliff Stability](#activity-cliff-stability)
- [Post-hoc Explanation Methods](#post-hoc-explanation-methods)
  - [CF-GNNExplainer](#cf-gnnexplainer)
  - [T-GNNExplainer](#t-gnnexplainer)
  - [CIDER Diagnostics](#cider-diagnostics)
- [Splitting Strategies](#splitting-strategies)
- [Configuration Reference](#configuration-reference)
- [CLI Reference](#cli-reference)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

---

## Overview

Modern drug discovery increasingly relies on machine learning to predict molecular properties, identify drug-target interactions, and prioritize drug candidates. Graph Neural Networks (GNNs) have emerged as particularly effective for these tasks because they naturally represent molecular structures as graphs of atoms and bonds.

However, GNNs are often treated as black boxes — they produce predictions but provide little insight into **why** a particular molecule is predicted to be active, toxic, or suitable as a drug candidate. This lack of transparency is a critical barrier to adoption in pharmaceutical research, where scientific reasoning and regulatory compliance demand interpretability.

**InterGNN** addresses this challenge by integrating multiple complementary interpretability methods directly into the GNN architecture:

- **Intrinsic interpretability** through prototype-based reasoning, motif extraction, and concept-aligned latent spaces
- **Post-hoc explanations** through counterfactual perturbations, sufficient subgraph identification, and causal invariance testing
- **Stability guarantees** ensuring explanations remain consistent under small input perturbations and across structurally similar molecules (activity cliffs)

The framework supports both **molecular property prediction** (e.g., toxicity, solubility, bioactivity) and **drug-target affinity** (DTA) prediction, with specialized encoders for both molecular graphs and protein target graphs.

---

## Key Features

| Category | Feature | Description |
|----------|---------|-------------|
| **Encoders** | Molecular GNN Encoder | Edge-aware GINEConv with chirality features, ~78-dim atom + ~14-dim bond feature vectors |
| | Target GNN Encoder | Multi-head GATConv for residue-level protein graphs with positional encodings |
| | Cross-Attention Fusion | Atom-residue interaction via scaled dot-product cross-attention for DTA tasks |
| **Interpretability** | PAGE Prototypes | Case-based classification via learned class-specific prototypes with pull/push/diversity losses |
| | MAGE Motifs | Differentiable motif mask generation using Gumbel-sigmoid with sparsity and connectivity regularization |
| | Concept Whitening | ZCA whitening + learnable rotation for axis-aligned concept interpretability |
| | Stability Regularization | Augmentation robustness + activity cliff consistency enforcement |
| **Explainers** | CF-GNNExplainer | Counterfactual explanations via minimal edge perturbation that flips predictions |
| | T-GNNExplainer | Sufficient subgraph identification preserving original predictions |
| | CIDER Diagnostics | Causal invariance testing across data environments |
| **Data** | 6 Benchmark Datasets | MUTAG, Tox21, ClinTox, QM9, Davis, SIDER |
| | Molecule Standardization | Tautomer canonicalization, charge neutralization, stereo handling, InChIKey deduplication |
| | Activity Cliff Detection | Fingerprint similarity-based cliff pair identification and tagging |
| | Splitting Strategies | Scaffold, cold-target, cold-drug, temporal, and stratified random splits |
| **Training** | Two-Phase Pipeline | Phase 1: pre-train encoders; Phase 2: joint fine-tuning with all interpretability losses |
| | Multi-Objective Loss | 9 configurable loss components with independent λ weights |
| **Evaluation** | Comprehensive Metrics | Predictive (ROC-AUC, CI), faithfulness (deletion/insertion AUC), stability (Jaccard), chemical validity, causal invariance, statistical tests |
| **Visualization** | Molecule Saliency | Atom/bond importance coloring via RDKit |
| | HTML Dashboard | Batch explanation reports with interactive exploration |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         InterGNN Pipeline                           │
│                                                                     │
│  ┌──────────────┐     ┌────────────────────┐     ┌──────────────┐  │
│  │ SMILES Input  │────▶│  Standardization   │────▶│ Featurization │  │
│  └──────────────┘     │  - Tautomer canon.  │     │ - Atom feats  │  │
│                       │  - Charge neutral.  │     │ - Bond feats  │  │
│                       │  - Stereo handling  │     │ - 2D/3D coords│  │
│                       └────────────────────┘     └───────┬──────┘  │
│                                                          │         │
│  ┌──────────────┐     ┌────────────────────┐             │         │
│  │Protein Seq/  │────▶│ Protein Graph      │             │         │
│  │Structure     │     │  - Residue features │             │         │
│  └──────────────┘     │  - k-NN / contact   │             │         │
│                       └────────┬───────────┘             │         │
│                                │                         │         │
│                                ▼                         ▼         │
│                  ┌──────────────────────┐  ┌──────────────────────┐│
│                  │  Target GNN Encoder  │  │ Molecular GNN Encoder ││
│                  │  (GATConv, 3 layers, │  │ (GINEConv, 4 layers, ││
│                  │   multi-head attn)   │  │  edge-aware MPNN)    ││
│                  └──────────┬───────────┘  └──────────┬───────────┘│
│                             │                         │            │
│                             ▼                         ▼            │
│                  ┌──────────────────────────────────────┐          │
│                  │       Cross-Attention Fusion          │          │
│                  │  Q = W_q · mol_nodes                  │          │
│                  │  K = W_k · target_nodes               │          │
│                  │  V = W_v · target_nodes               │          │
│                  │  A = softmax(QK^T / √d)               │          │
│                  │  z_fused = [z_m ‖ z_p ‖ pool(AV)]    │          │
│                  └──────────────────┬───────────────────┘          │
│                                     │                              │
│              ┌──────────────────────┼──────────────────────┐       │
│              ▼                      ▼                      ▼       │
│  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐   │
│  │ Prototype Layer   │ │  Motif Generator  │ │Concept Whitening │   │
│  │ - L2 distances    │ │ - Gumbel-sigmoid  │ │ - ZCA transform  │   │
│  │ - Pull/Push loss  │ │ - Sparsity loss   │ │ - Axis alignment │   │
│  │ - Case-based      │ │ - Connectivity    │ │ - Decorrelation  │   │
│  └────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘   │
│           │                    │                     │              │
│           └────────────────────┼─────────────────────┘              │
│                                ▼                                    │
│                    ┌──────────────────────┐                         │
│                    │     Task Head        │                         │
│                    │  Classification/     │                         │
│                    │  Regression MLP      │                         │
│                    └──────────┬───────────┘                         │
│                               │                                    │
│                               ▼                                    │
│                    ┌──────────────────────┐                         │
│                    │    Predictions +     │                         │
│                    │    Explanations      │                         │
│                    └─────────────────────┘                         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/HarshalLoya/InterGNN.git
cd InterGNN

# Install in editable mode with all extras
pip install -e ".[vis,dev]"
```

### Dependencies

#### Core (installed automatically)

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥ 2.0.0 | Deep learning framework |
| `torch-geometric` | ≥ 2.4.0 | Graph neural network library |
| `rdkit` | ≥ 2023.3.1 | Cheminformatics toolkit |
| `numpy` | ≥ 1.24.0 | Numerical computing |
| `scipy` | ≥ 1.10.0 | Scientific computing |
| `pandas` | ≥ 2.0.0 | Data manipulation |
| `scikit-learn` | ≥ 1.2.0 | Machine learning utilities |
| `matplotlib` | ≥ 3.7.0 | Plotting and visualization |
| `pyyaml` | ≥ 6.0 | YAML configuration parsing |
| `tqdm` | ≥ 4.65.0 | Progress bars |

#### Optional Visualization (`pip install -e ".[vis]"`)

| Package | Purpose |
|---------|---------|
| `plotly` | Interactive plots |
| `py3Dmol` | 3D molecular visualization |
| `seaborn` | Statistical plotting |
| `ipywidgets` | Jupyter widget support |

#### Development (`pip install -e ".[dev]"`)

| Package | Purpose |
|---------|---------|
| `pytest` | Testing framework |
| `pytest-cov` | Coverage reporting |
| `black` | Code formatting |
| `ruff` | Linting |
| `mypy` | Type checking |

---

## Quick Start

### 1. Create a Configuration File

InterGNN uses YAML configuration files to control all aspects of the pipeline. Create a `config.yaml`:

```yaml
# config.yaml — InterGNN Configuration

data:
  dataset_name: tox21           # Dataset to use
  split_method: scaffold        # Splitting strategy
  batch_size: 32                # Mini-batch size
  frac_train: 0.8               # Training fraction
  frac_val: 0.1                 # Validation fraction
  frac_test: 0.1                # Test fraction
  detect_cliffs: true           # Enable activity cliff detection
  cliff_sim_threshold: 0.9      # Similarity threshold for cliffs
  cliff_act_threshold: 1.0      # Activity threshold for cliffs
  compute_concepts: true        # Compute SMARTS concept vectors
  seed: 42                      # Random seed

model:
  hidden_dim: 256               # Latent dimension for all layers
  num_mol_layers: 4             # GINEConv message passing layers
  num_target_layers: 3          # GATConv layers for protein
  num_attn_heads: 4             # Cross-attention heads
  task_type: classification     # 'classification' or 'regression'
  num_tasks: 12                 # Number of prediction targets (Tox21 has 12)
  dropout: 0.2                  # Dropout rate
  use_target: false             # Enable drug-target interaction mode
  fusion_type: cross_attention  # 'cross_attention' or 'bilinear'
  readout: attention            # 'attention', 'mean', or 'sum'

interpretability:
  use_prototypes: true          # Enable PAGE prototype layer
  num_prototypes_per_class: 5   # Prototypes per class
  prototype_activation: log     # 'log' or 'linear' distance-to-similarity

  use_motifs: true              # Enable MAGE motif generator
  num_motifs: 8                 # Number of motif patterns to discover
  motif_temperature: 0.5        # Gumbel-sigmoid temperature
  motif_sparsity_target: 0.3    # Target atom selection ratio

  use_concept_whitening: true   # Enable concept whitening
  num_concepts: 30              # Number of aligned concept dimensions

  use_stability: false          # Enable stability regularizer

loss:
  lambda_pred: 1.0              # Prediction loss weight
  lambda_pull: 0.1              # Prototype pull loss
  lambda_push: 0.05             # Prototype push loss
  lambda_diversity: 0.01        # Prototype diversity loss
  lambda_motif: 0.1             # Motif sparsity loss
  lambda_connectivity: 0.05     # Motif connectivity loss
  lambda_concept: 0.1           # Concept alignment loss
  lambda_decorrelation: 0.01    # Concept decorrelation loss
  lambda_stability: 0.05        # Explanation stability loss

training:
  pretrain_epochs: 50           # Phase 1 epochs
  finetune_epochs: 100          # Phase 2 epochs
  learning_rate: 0.001          # Initial learning rate
  weight_decay: 0.00001         # L2 regularization
  lr_scheduler: cosine          # 'cosine', 'step', or 'plateau'
  warmup_steps: 500             # Learning rate warmup
  gradient_clip: 1.0            # Gradient clipping norm
  early_stopping_patience: 15   # Stop after N epochs without improvement
  checkpoint_dir: checkpoints   # Directory for model checkpoints
  log_interval: 10              # Log every N epochs
  device: auto                  # 'auto', 'cuda', or 'cpu'
  seed: 42
```

### 2. Training

```bash
# Train with two-phase pipeline
inter-gnn train --config config.yaml

# Save resolved config for reproducibility
inter-gnn train --config config.yaml --save-config resolved_config.yaml
```

### 3. Evaluation

```bash
# Evaluate on test set
inter-gnn evaluate --config config.yaml --checkpoint checkpoints/finetune_best.pt
```

### 4. Generating Explanations

```bash
# Explain specific molecules
inter-gnn explain \
    --config config.yaml \
    --checkpoint model.pt \
    --smiles "CC(=O)Oc1ccccc1C(=O)O" "c1ccc2[nH]c3ccccc3c2c1" \
    --output explanations.json
```

### 5. Explanation Dashboard

```bash
# Generate interactive HTML dashboard for the test set
inter-gnn dashboard \
    --config config.yaml \
    --checkpoint model.pt \
    --output dashboard_report/ \
    --max-samples 200
```

---

## Python API

### Data Pipeline

```python
from inter_gnn.data.standardize import standardize_smiles, batch_standardize
from inter_gnn.data.featurize import smiles_to_graph, smiles_to_3d_graph
from inter_gnn.data.protein import ProteinGraphBuilder
from inter_gnn.data.concepts import concept_vector, CONCEPT_LIBRARY
from inter_gnn.data.cliffs import detect_activity_cliffs, tag_cliff_molecules
from inter_gnn.data.splits import scaffold_split, cold_target_split

# ── Standardize a molecule ──
clean_smi = standardize_smiles("CC(=O)Oc1ccccc1C(=O)O")

# ── Convert SMILES to molecular graph ──
graph = smiles_to_graph(clean_smi)
# graph.x     → (N_atoms, 55) atom features
# graph.edge_index → (2, E) edge connectivity
# graph.edge_attr  → (E, 14) bond features

# ── 3D graph with distance features ──
graph_3d = smiles_to_3d_graph(clean_smi, n_conformers=5)

# ── Protein graph construction ──
builder = ProteinGraphBuilder(k=10, contact_threshold=8.0)
protein_graph = builder.from_sequence("MKTLLILAVFCLGFASS...")

# ── Compute concept vector ──
concepts = concept_vector(clean_smi)  # (30,) binary vector
print(f"Concepts: {concepts.sum()} active out of {len(CONCEPT_LIBRARY)}")

# ── Activity cliff detection ──
cliffs = detect_activity_cliffs(
    smiles_list=["CCO", "CCCO", "c1ccccc1"],
    activities=[5.0, 8.5, 3.0],
    sim_threshold=0.9,
    act_threshold=1.0,
)

# ── Scaffold splitting ──
train_idx, val_idx, test_idx = scaffold_split(
    smiles_list=all_smiles,
    frac_train=0.8, frac_val=0.1, frac_test=0.1,
)
```

### Model Construction

```python
import torch
from inter_gnn.models.core_model import InterGNN
from inter_gnn.models.encoders import MolecularGNNEncoder, TargetGNNEncoder
from inter_gnn.models.attention import CrossAttentionFusion
from inter_gnn.models.task_heads import TaskHead

# ── Build the full model ──
model = InterGNN(
    atom_feat_dim=55,
    bond_feat_dim=14,
    hidden_dim=256,
    num_mol_layers=4,
    task_type="classification",
    num_tasks=12,
    dropout=0.2,
    use_target=False,        # False for molecular property prediction
    readout="attention",
)

# ── Forward pass ──
output = model(
    x=graph.x,
    edge_index=graph.edge_index,
    edge_attr=graph.edge_attr,
    batch=torch.zeros(graph.x.shape[0], dtype=torch.long),
)

print(output["prediction"].shape)   # (1, 12)
print(output["mol_node_emb"].shape) # (N_atoms, 256)
print(output["mol_graph_emb"].shape)# (1, 256)

# ── Drug-Target Affinity mode ──
dta_model = InterGNN(
    atom_feat_dim=55,
    bond_feat_dim=14,
    residue_feat_dim=42,
    hidden_dim=256,
    task_type="regression",
    num_tasks=1,
    use_target=True,             # Enable cross-attention fusion
    fusion_type="cross_attention",
)

dta_output = dta_model(
    x=mol_graph.x,
    edge_index=mol_graph.edge_index,
    edge_attr=mol_graph.edge_attr,
    batch=mol_batch,
    x_target=protein_graph.x,
    edge_index_target=protein_graph.edge_index,
    batch_target=protein_batch,
)
# dta_output["attention_weights"] → atom-residue attention for interpretability
```

### Training Pipeline

```python
from inter_gnn.training.config import InterGNNConfig
from inter_gnn.training.trainer import InterGNNTrainer
from inter_gnn.data.datamodule import InterGNNDataModule

# ── Load configuration ──
config = InterGNNConfig.from_yaml("config.yaml")

# ── Or build programmatically ──
config = InterGNNConfig()
config.data.dataset_name = "tox21"
config.data.split_method = "scaffold"
config.model.hidden_dim = 256
config.model.task_type = "classification"
config.model.num_tasks = 12
config.interpretability.use_prototypes = True
config.interpretability.use_motifs = True
config.training.pretrain_epochs = 50
config.training.finetune_epochs = 100

# ── Build data pipeline ──
dm = InterGNNDataModule(config)
dm.prepare_data()
dm.setup()

# ── Train ──
trainer = InterGNNTrainer(config)
history = trainer.fit(dm.train_dataloader(), dm.val_dataloader())

# history is a list of dicts with per-epoch metrics:
# [{"phase": "pretrain", "epoch": 1, "prediction": 0.42, "epoch_total": 0.42, ...}, ...]

# ── Save config for reproducibility ──
config.to_yaml("experiment_config.yaml")
```

### Generating Explanations

```python
from inter_gnn.explainers.cf_explainer import CFGNNExplainer
from inter_gnn.explainers.t_explainer import TGNNExplainer
from inter_gnn.explainers.cider import CIDERDiagnostics

# ── Gradient-based node importance ──
importance = model.get_node_importance(
    graph.x, graph.edge_index, graph.edge_attr, batch
)
# importance: (N_atoms,) — higher = more important for prediction

# ── Counterfactual explanations ──
cf_explainer = CFGNNExplainer(
    model=model,
    lr=0.01,
    num_iterations=500,
    edge_loss_weight=1.0,
    prediction_loss_weight=2.0,
)

cf_result = cf_explainer.explain(graph, target_class=0)
# cf_result["success"]       → True if prediction was flipped
# cf_result["num_edits"]     → Number of edges removed
# cf_result["edge_mask"]     → Continuous edge importance mask
# cf_result["removed_edges"] → Indices of removed edges

# ── Sufficient subgraph explanations ──
t_explainer = TGNNExplainer(
    model=model,
    lr=0.01,
    num_iterations=300,
    node_mask_weight=0.1,
    edge_mask_weight=1.0,
)

t_result = t_explainer.explain(graph, threshold=0.5)
# t_result["fidelity"]         → How well subgraph preserves prediction
# t_result["important_nodes"]  → Atom indices in the sufficient subgraph
# t_result["important_edges"]  → Edge indices in the sufficient subgraph

# ── CIDER causal invariance diagnostics ──
cider = CIDERDiagnostics(model=model, k=10)
diagnostics = cider.run_full_diagnostics({
    "scaffold_A": data_list_env_a,
    "scaffold_B": data_list_env_b,
})
# diagnostics["invariance"]["overall_invariance"] → cross-env consistency
# diagnostics["spurious_detection"]["fraction_spurious"] → fraction flagged
```

### Evaluation API

```python
from inter_gnn.evaluation.predictive import (
    compute_classification_metrics,
    compute_regression_metrics,
)
from inter_gnn.evaluation.faithfulness import deletion_auc, insertion_auc, sufficiency_score
from inter_gnn.evaluation.stability_metrics import jaccard_stability, cliff_consistency
from inter_gnn.evaluation.chemical_validity import explanation_validity_report
from inter_gnn.evaluation.causal import invariance_violation_rate
from inter_gnn.evaluation.statistical import paired_bootstrap_test

# ── Predictive metrics ──
cls_metrics = compute_classification_metrics(predictions, targets)
# {"roc_auc": 0.89, "pr_auc": 0.82, "accuracy": 0.85, "f1_score": 0.78, "mcc": 0.70}

reg_metrics = compute_regression_metrics(predictions, targets)
# {"rmse": 0.65, "mae": 0.48, "r2": 0.91, "pearson_r": 0.96, "ci": 0.88}

# ── Faithfulness ──
del_auc = deletion_auc(model, graph, importance, num_steps=10)
ins_auc = insertion_auc(model, graph, importance, num_steps=10)
suff = sufficiency_score(model, graph, node_mask)

# ── Stability ──
consistency = cliff_consistency(
    explanations=[imp_a.numpy(), imp_b.numpy()],
    cliff_pairs=[(0, 1)],
    top_k=5,
)

# ── Statistical significance ──
result = paired_bootstrap_test(metric_a_samples, metric_b_samples, num_bootstrap=10000)
# result["p_value"], result["significant_at_005"], result["ci_95_lower"], result["ci_95_upper"]
```

### Visualization API

```python
from inter_gnn.visualization.molecule_viz import render_atom_importance, batch_render_explanations
from inter_gnn.visualization.prototype_viz import plot_prototype_gallery, plot_prototype_distances
from inter_gnn.visualization.motif_viz import plot_motif_activation_heatmap, render_motif_overlay
from inter_gnn.visualization.concept_viz import plot_concept_activations, plot_concept_comparison
from inter_gnn.visualization.counterfactual_viz import render_counterfactual_comparison
from inter_gnn.visualization.dashboard import ExplanationDashboard

# ── Atom saliency ──
render_atom_importance(
    smiles="CC(=O)Oc1ccccc1C(=O)O",
    atom_importance=importance.numpy(),
    save_path="aspirin_importance.svg",
    cmap="RdYlGn_r",
    title="Aspirin — Atom Importance",
)

# ── Prototype gallery ──
plot_prototype_gallery(
    prototype_embeddings=proto_emb,
    nearest_smiles={0: ["CCO", "CCCO"], 1: ["c1ccccc1", "c1ccncc1"]},
    save_path="prototypes.png",
)

# ── Motif heatmap ──
plot_motif_activation_heatmap(
    motif_activations=activations_matrix,
    motif_names=["Ring", "Halogen", "Amine", "Carbonyl"],
    save_path="motifs.png",
)

# ── HTML dashboard ──
dashboard = ExplanationDashboard("output_dir/", title="Tox21 Explanations")
for smi, pred, imp in zip(smiles_list, predictions, importances):
    dashboard.add_entry(smiles=smi, prediction=pred, atom_importance=imp)
dashboard.generate()  # → output_dir/dashboard.html
```

---

## Module Reference

### Data & Preprocessing (`inter_gnn/data/`)

#### `standardize.py` — Molecule Standardization

Full cheminformatics standardization pipeline:
- **Tautomer canonicalization** — Resolves tautomeric ambiguity using RDKit's `TautomerCanonicalizer`
- **Charge neutralization** — Removes unnecessary formal charges where applicable
- **Stereochemistry handling** — Normalizes and validates chiral centers
- **Fragment stripping** — Removes salts and small fragments, keeping largest component
- **InChIKey deduplication** — Removes exact chemical duplicates using InChIKey hashing
- **Quality filters** — Enforces minimum heavy atom count and maximum molecular weight

```python
clean_smiles = standardize_smiles("CC(=O)Oc1ccccc1C(=O)O")
cleaned_batch = batch_standardize(["CCO", "CCCO", "invalid_smiles"])
deduplicated = deduplicate_smiles(smiles_list)
```

#### `featurize.py` — Molecular Graph Construction

Converts SMILES strings into PyTorch Geometric `Data` objects:

- **Atom features (~55 dimensions):**
  - Atomic number (one-hot, 28 types including common organic + metals)
  - Degree (0–6)
  - Formal charge (-2 to +2)
  - Hybridization (sp, sp², sp³, sp³d, sp³d²)
  - Chirality (none, R, S)
  - Aromaticity, ring membership, number of hydrogens, radical electrons

- **Bond features (~14 dimensions):**
  - Bond type (single, double, triple, aromatic)
  - Stereo configuration (none, E, Z, any)
  - Conjugation, ring membership

- **3D support:** Optional conformer generation with RBF distance expansion for 3D-aware message passing

#### `protein.py` — Protein Graph Construction

Builds residue-level graphs from protein sequences or 3D structures:
- **k-NN sequence graph:** Connects nearby residues based on feature similarity
- **Contact graph:** Connects residues with C-α distance < threshold (from PDB coordinates)
- **Residue features:** One-hot amino acid encoding + physicochemical properties (hydrophobicity, charge, volume, polarity) + sinusoidal positional encodings

#### `concepts.py` — SMARTS Concept Library

Curated library of ~30 SMARTS patterns organized by category:
- **Pharmacophores:** Hydrogen bond donors/acceptors, aromatic rings, positive/negative ionizable groups
- **Functional groups:** Hydroxyl, carboxyl, amine, amide, ester, sulfone, nitro, halogen, phosphate
- **Toxicophores:** PAINS alerts, reactive groups, Michael acceptors
- **Drug-likeness:** Ring systems, rotatable bonds, aromatic nitrogen

#### `cliffs.py` — Activity Cliff Detection

Identifies molecule pairs with high structural similarity but large activity differences:
- Fingerprint computation (Morgan/RDKit/MACCS)
- Tanimoto similarity filtering
- Activity difference thresholding
- Cliff pair scoring and molecule tagging

#### `splits.py` — Dataset Splitting

- **Scaffold split:** Bemis-Murcko scaffold decomposition for chemically meaningful train/val/test separation
- **Cold-target / cold-drug split:** Ensures unseen targets or drugs in test set for DTA generalization
- **Temporal split:** Chronological data ordering for realistic prospective evaluation
- **Stratified random split:** Preserves label distribution across splits
- **k-Fold cross-validation:** Scaffold-aware or random k-fold iterator

#### `datasets.py` — Benchmark Dataset Loaders

Unified loader interface for 9 benchmark datasets:

| Dataset | Type | Tasks | Source |
|---------|------|-------|--------|
| MUTAG | Graph Classification | 1 | TUDatasets |
| Tox21 | Mol. Classification | 12 | MoleculeNet |
| ClinTox | Mol. Classification | 2 | MoleculeNet |
| QM9 | Mol. Regression | 19 | MoleculeNet |
| Davis | DTA Regression | 1 | TDC |
| SIDER | Mol. Classification | 27 | MoleculeNet |

#### `datamodule.py` — DataModule Wrapper

Orchestrates the full data pipeline:
1. Dataset loading via registry
2. Splitting with configured strategy
3. Activity cliff detection and tagging
4. Concept vector computation
5. DataLoader creation with proper batching

---

### Core Model (`inter_gnn/models/`)

#### `encoders.py` — GNN Encoders

**MolecularGNNEncoder:**
- L layers of GINEConv with edge-feature-aware message passing
- BatchNorm + GELU activation + dropout per layer
- Residual connections between layers
- Configurable readout: attention-weighted, mean, or sum pooling

**TargetGNNEncoder:**
- L layers of multi-head GATConv for residue graphs
- Attention-based residue interaction learning
- Same readout options as molecular encoder

**AttentionReadout:**
- Learnable gate network for graph-level attention-weighted pooling
- Softmax normalization within each graph in the batch

#### `attention.py` — Cross-Attention Fusion

**CrossAttentionFusion:**
- Multi-head scaled dot-product attention between atom (query) and residue (key/value) embeddings
- Scatter-to-padded batch conversion for variable-size graphs
- Residual connection + LayerNorm
- Fusion projection: `[z_mol ‖ z_target ‖ pool(attended)]` → hidden_dim

**BilinearFusion:**
- Ablation alternative: `z = W₁z_m + W₂z_p + z_m^T W_b z_p + b`
- Simpler but less expressive than cross-attention

#### `task_heads.py` — Prediction Heads

**ClassificationHead:** 3-layer MLP with sigmoid (multi-label) or softmax (multi-class)
**RegressionHead:** 3-layer MLP for continuous output
**TaskHead():** Factory function dispatching to the appropriate head

#### `core_model.py` — InterGNN Model

Unified model wiring all components:
- Molecular encoder → (optional) Target encoder → (optional) Fusion → Interpretability hooks → Task head
- Supports molecule-only mode and drug-target interaction mode
- Gradient-based node importance extraction via `get_node_importance()`

---

### Intrinsic Interpretability (`inter_gnn/interpretability/`)

#### `prototypes.py` — PAGE Prototypes

- Learnable class-specific prototype vectors in latent space
- L2 distance-based similarity with log or linear activation
- Fixed prototype-to-class mapping with learned classifier weights
- **Pull loss:** Minimize sample-to-same-class prototype distance
- **Push loss:** Maximize sample-to-different-class prototype distance
- **Diversity loss:** Spread same-class prototypes apart

#### `motifs.py` — MAGE Motifs

- Per-node mask predictor: MLP → K motif scores per atom
- Gumbel-sigmoid for differentiable discrete mask sampling
- Per-motif embeddings via masked-weighted pooling
- **Sparsity loss:** Target fraction of active atoms
- **Connectivity loss:** Penalize isolated masked atoms
- **MotifExtractor:** Post-hoc discrete motif extraction

#### `concept_whitening.py` — Concept Whitening

- ZCA whitening of latent representations using running statistics
- Learnable rotation matrix for concept axis alignment
- Per-concept linear probes for alignment supervision
- **Alignment loss:** BCE between concept probe predictions and concept labels
- **Decorrelation loss:** Off-diagonal correlation penalty

#### `stability.py` — Explanation Stability

- **Augmentation stability:** Feature masking + edge dropping → cosine similarity of explanations
- **Cliff stability:** L2 distance between normalized explanations of cliff pairs
- Combined weighted loss for training regularization

---

### Post-hoc Explainers (`inter_gnn/explainers/`)

#### `cf_explainer.py` — CF-GNNExplainer

Counterfactual explanations via learned edge perturbation:
- Optimizes continuous edge mask to minimize: λ_pred × prediction_change + λ_edge × edit_size
- Sigmoid temperature annealing for mask discretization
- Finds minimal structural edits that flip the model's prediction

#### `t_explainer.py` — T-GNNExplainer

Sufficient subgraph identification:
- Jointly learns node and edge masks
- Optimizes: prediction_preservation − λ × mask_size + entropy_regularization
- Identifies the smallest subgraph that reproduces the full prediction

#### `cider.py` — CIDER Diagnostics

Causal invariance testing:
- Groups molecules by environments (scaffolds, assays, etc.)
- Computes top-k Jaccard invariance of important atoms across environments
- Flags potentially spurious features (high importance in one env only)
- Full diagnostic report with per-molecule and per-environment statistics

---

### Training Pipeline (`inter_gnn/training/`)

#### `losses.py` — Loss Functions

**PredictionLoss:** BCE (classification) or MSE (regression) with NaN masking for multi-task

**TotalLoss:** Combined objective with 9 independently weighted components:
```
L_total = λ₁·L_pred + λ₂·L_pull + λ₃·L_push + λ₄·L_diversity
        + λ₅·L_motif + λ₆·L_connectivity + λ₇·L_concept
        + λ₈·L_decorrelation + λ₉·L_stability
```

#### `trainer.py` — Two-Phase Trainer

- **Phase 1 (Pre-training):** Encoders + task head with prediction loss only
- **Phase 2 (Fine-tuning):** All modules with full multi-objective loss at 0.1× LR
- AdamW optimizer with configurable weight decay
- Cosine / StepLR / ReduceLROnPlateau scheduling
- Gradient clipping, early stopping, checkpointing

#### `callbacks.py` — Training Callbacks

- **EarlyStopping:** Configurable patience, min_delta, and mode (min/max)
- **ModelCheckpoint:** Top-k checkpoint saving with automatic cleanup
- **ExplainerMonitor:** Periodic gradient-based importance diagnostics
- **CallbackManager:** Unified callback lifecycle management

#### `config.py` — Configuration

Hierarchical dataclass configuration with YAML serialization:
- `DataConfig`, `ModelConfig`, `InterpretabilityConfig`, `LossConfig`, `TrainingConfig`
- `InterGNNConfig.from_yaml()` and `.to_yaml()` methods
- Sensible defaults for all parameters

---

### Evaluation Metrics (`inter_gnn/evaluation/`)

#### `predictive.py` — Predictive Performance

| Metric | Type | Range | Ideal |
|--------|------|-------|-------|
| ROC-AUC | Classification | [0, 1] | 1.0 |
| PR-AUC | Classification | [0, 1] | 1.0 |
| Accuracy | Classification | [0, 1] | 1.0 |
| F1 Score | Classification | [0, 1] | 1.0 |
| MCC | Classification | [-1, 1] | 1.0 |
| RMSE | Regression | [0, ∞) | 0.0 |
| MAE | Regression | [0, ∞) | 0.0 |
| R² | Regression | (-∞, 1] | 1.0 |
| Pearson r | Regression | [-1, 1] | 1.0 |
| Spearman ρ | Regression | [-1, 1] | 1.0 |
| CI | Regression | [0, 1] | 1.0 |

#### `faithfulness.py` — Explanation Faithfulness

- **Deletion AUC:** Progressive removal of important atoms → prediction degradation (lower = better)
- **Insertion AUC:** Progressive insertion of important atoms → prediction recovery (higher = better)
- **Sufficiency:** Does the explanation subgraph alone reproduce the prediction?
- **Necessity:** Does removing the explanation change the prediction?

#### `stability_metrics.py` — Explanation Stability

- **Jaccard stability:** Overlap of important atom sets under augmentation
- **Cliff consistency:** Top-k atom overlap between activity cliff pairs
- **Rank correlation stability:** Spearman ρ between importance vectors

#### `chemical_validity.py` — Chemical Validity

- Valence checking for generated molecular substructures
- SMARTS pattern match rates across molecule sets
- Property distribution analysis (MW, LogP, heavy atoms)

#### `causal.py` — Causal Invariance

- **Invariance violation rate:** Fraction of predictions changing across environments
- **Environment alignment score:** Cross-environment top-k feature Jaccard

#### `statistical.py` — Statistical Tests

- **Paired bootstrap test:** P-value with 95% CI for model comparison
- **Randomization test:** Permutation-based significance testing

---

### Visualization (`inter_gnn/visualization/`)

| Module | Output |
|--------|--------|
| `molecule_viz.py` | SVG/PNG with atoms colored by importance score |
| `prototype_viz.py` | Gallery of nearest training examples per prototype; distance histogram |
| `motif_viz.py` | Activation heatmaps across molecules; multi-color motif overlays |
| `concept_viz.py` | Horizontal bar charts of concept activations; multi-sample comparisons |
| `counterfactual_viz.py` | Side-by-side original vs. counterfactual with edit annotations |
| `dashboard.py` | HTML report with summary statistics + per-molecule explanation table |

---

## Supported Datasets

| Dataset | Domain | Type | Tasks | Molecules | Source |
|---------|--------|------|-------|-----------|--------|
| **MUTAG** | Mutagenicity | Graph classification | 1 | 188 | TUDatasets |
| **Tox21** | Toxicology | Multi-label classification | 12 | 7,831 | MoleculeNet |
| **ClinTox** | Clinical trials | Classification | 2 | 1,484 | MoleculeNet |
| **QM9** | Quantum chemistry | Regression | 19 | 130,831 | MoleculeNet |
| **Davis** | Kinase binding | DTA regression | 1 | 30,056 pairs | TDC |
| **KIBA** | Kinase binding | DTA regression | 1 | 118,254 pairs | TDC |
| **BindingDB** | Drug-target | DTA regression | 1 | 39,747 pairs | TDC |
| **SIDER** | Side effects | Multi-label classification | 27 | 1,427 | MoleculeNet |
| **SynLethDB** | Synthetic lethality | Link prediction | 1 | 19,667 pairs | Custom |

---

## Two-Phase Training Strategy

InterGNN uses a principled two-phase training approach:

### Phase 1: Pre-training (Encoders + Task Head)

**Goal:** Establish high-quality molecular representations before adding interpretability constraints.

- Only the molecular encoder, (optional) target encoder, and task head are active
- Only prediction loss is used (`L_pred`)
- Full learning rate with cosine annealing
- Early stopping based on validation loss

### Phase 2: Joint Fine-tuning (All Modules)

**Goal:** Refine representations to simultaneously support accurate predictions AND interpretable explanations.

- Prototype layer, motif generator, and concept whitening are attached
- All 9 loss components are active with configurable λ weights
- Learning rate reduced to 0.1× of Phase 1
- Separate early stopping and checkpointing

**Rationale:** Joint training from scratch can lead to degenerate solutions where interpretability modules dominate the loss landscape before useful representations are learned. Pre-training ensures a strong foundation.

---

## Interpretability Deep Dive

### PAGE Prototypes

Inspired by the **Prototype-based GNN Explanations (PAGE)** paradigm:

1. For each class, maintain K learnable prototype vectors in latent space
2. For each input graph, compute L2 distance to all prototypes
3. Classify by weighted distance to class prototypes
4. Explain: "This molecule is predicted toxic because it is most similar to prototype #3"

**Training losses:**
- **Pull:** Each sample is attracted to its nearest same-class prototype
- **Push:** Each sample is repelled from its nearest different-class prototype
- **Diversity:** Same-class prototypes are encouraged to spread apart

### MAGE Motifs

Inspired by **Molecular Automatic Generation of Explanations (MAGE)**:

1. For each atom, predict K motif membership scores
2. Apply Gumbel-sigmoid for differentiable discrete masks
3. Pool masked node embeddings to get per-motif representations
4. Regularize for sparsity (target fraction of active atoms) and connectivity (motif atoms should be bonded)

**Post-hoc extraction:** After training, `MotifExtractor` converts continuous masks to discrete substructures.

### Concept Whitening

Inspired by **Concept Whitening** literature:

1. **ZCA whitening:** Decorrelate latent dimensions using running covariance statistics
2. **Rotation learning:** Learn a rotation matrix R such that dimension i activates when concept i is present
3. **Concept supervision:** Binary concept labels (from SMARTS matching) guide the rotation

**Result:** Each latent dimension has a human-interpretable meaning (e.g., "dimension 3 ≈ presence of hydroxyl group").

### Activity Cliff Stability

Activity cliffs are molecule pairs with high structural similarity but large activity differences. They are particularly challenging for GNNs because similar inputs should produce similar representations, yet the outputs differ dramatically.

**Stability regularization ensures:**
1. Explanations for augmented inputs are consistent with originals (augmentation robustness)
2. Explanations for cliff pairs share common structural features (cliff consistency)

---

## Post-hoc Explanation Methods

### CF-GNNExplainer

**Question answered:** "What is the minimal change to this molecule that would flip the prediction?"

- Learns a continuous edge mask via gradient-based optimization
- Balances prediction change (maximize) against edit size (minimize)
- Optimal counterfactual reveals which bonds are essential

### T-GNNExplainer

**Question answered:** "What is the smallest subgraph sufficient to reproduce this prediction?"

- Jointly learns node and edge importance masks
- Entropy regularization pushes masks toward binary (fully in/out)
- Fidelity metric measures how well the subgraph preserves the original prediction

### CIDER Diagnostics

**Question answered:** "Are the model's explanations capturing causal features or spurious correlations?"

- Groups data into environments (scaffold clusters, assay types, etc.)
- Checks if top-k important atoms are consistent across environments
- Flags molecules where explanations are environment-specific (potentially spurious)

---

## Splitting Strategies

| Strategy | Use Case | How It Works |
|----------|----------|-------------|
| **Scaffold Split** | General molecular ML | Groups by Bemis-Murcko scaffold; ensures structurally distinct train/test |
| **Cold-Target Split** | DTA generalization | Test targets are unseen during training |
| **Cold-Drug Split** | DTA generalization | Test drugs are unseen during training |
| **Temporal Split** | Prospective evaluation | Train on older data, test on newer data |
| **Stratified Random** | Balanced baselines | Random split preserving label distribution |

**Recommendation:** Use **scaffold split** for molecular property prediction and **cold-target split** for DTA tasks to best evaluate real-world generalization.

---

## Configuration Reference

All configuration parameters with defaults:

```yaml
data:
  dataset_name: mutag          # str: dataset identifier
  data_dir: null               # str | null: custom data directory
  split_method: scaffold       # str: scaffold | cold_target | temporal | random
  frac_train: 0.8              # float: training fraction
  frac_val: 0.1                # float: validation fraction
  frac_test: 0.1               # float: test fraction
  batch_size: 64               # int: mini-batch size
  num_workers: 0               # int: DataLoader worker processes
  detect_cliffs: false         # bool: enable cliff detection
  cliff_sim_threshold: 0.9     # float: Tanimoto similarity threshold
  cliff_act_threshold: 1.0     # float: activity difference threshold
  compute_concepts: false      # bool: compute SMARTS concept vectors
  seed: 42                     # int: random seed

model:
  atom_feat_dim: 55            # int: input atom feature dimension
  bond_feat_dim: 14            # int: input bond feature dimension
  residue_feat_dim: 42         # int: input residue feature dimension
  hidden_dim: 256              # int: shared hidden dimension
  num_mol_layers: 4            # int: GINEConv layers
  num_target_layers: 3         # int: GATConv layers
  num_attn_heads: 4            # int: attention heads
  task_type: classification    # str: classification | regression
  num_tasks: 1                 # int: number of prediction targets
  dropout: 0.2                 # float: dropout rate
  use_target: false            # bool: enable DTA mode
  fusion_type: cross_attention # str: cross_attention | bilinear
  readout: attention           # str: attention | mean | sum

interpretability:
  use_prototypes: false        # bool: enable prototype layer
  num_prototypes_per_class: 5  # int: prototypes per class
  prototype_activation: log    # str: log | linear
  use_motifs: false            # bool: enable motif generator
  num_motifs: 8                # int: motif patterns
  motif_temperature: 0.5       # float: Gumbel-sigmoid temperature
  motif_sparsity_target: 0.3   # float: target active fraction
  use_concept_whitening: false # bool: enable concept whitening
  num_concepts: 30             # int: aligned concept dimensions
  concept_momentum: 0.1        # float: running stats momentum
  use_stability: false         # bool: enable stability loss
  stability_mask_prob: 0.1     # float: feature mask probability
  stability_edge_drop_prob: 0.05 # float: edge drop probability

loss:
  lambda_pred: 1.0             # float: prediction loss weight
  lambda_pull: 0.1             # float: prototype pull weight
  lambda_push: 0.05            # float: prototype push weight
  lambda_diversity: 0.01       # float: prototype diversity weight
  lambda_motif: 0.1            # float: motif sparsity weight
  lambda_connectivity: 0.05    # float: motif connectivity weight
  lambda_concept: 0.1          # float: concept alignment weight
  lambda_decorrelation: 0.01   # float: concept decorrelation weight
  lambda_stability: 0.05       # float: explanation stability weight

training:
  pretrain_epochs: 50          # int: Phase 1 epochs
  finetune_epochs: 100         # int: Phase 2 epochs
  learning_rate: 0.001         # float: initial LR
  weight_decay: 0.00001        # float: L2 regularization
  lr_scheduler: cosine         # str: cosine | step | plateau
  warmup_steps: 500            # int: LR warmup steps
  gradient_clip: 1.0           # float: gradient clipping norm
  early_stopping_patience: 15  # int: early stopping patience
  checkpoint_dir: checkpoints  # str: checkpoint directory
  log_interval: 10             # int: logging frequency (epochs)
  eval_interval: 1             # int: validation frequency (epochs)
  device: auto                 # str: auto | cuda | cpu
  seed: 42                     # int: training seed
```

---

## CLI Reference

```
inter-gnn — Interpretable GNN for Drug Discovery

Usage:
  inter-gnn [--verbose] <command> [options]

Commands:
  train       Train the InterGNN model (two-phase pipeline)
  evaluate    Evaluate a trained model on the test set
  explain     Generate explanations for specific SMILES
  dashboard   Generate an interactive HTML explanation dashboard

Global Options:
  --verbose, -v    Enable debug logging

Train Options:
  --config PATH           YAML configuration file (required)
  --save-config PATH      Save resolved configuration

Evaluate Options:
  --config PATH           YAML configuration file (required)
  --checkpoint PATH       Model checkpoint file (required)

Explain Options:
  --config PATH           YAML configuration file (required)
  --checkpoint PATH       Model checkpoint file (required)
  --smiles SMILES [...]   One or more SMILES strings (required)
  --output PATH           Output JSON file for explanations

Dashboard Options:
  --config PATH           YAML configuration file (required)
  --checkpoint PATH       Model checkpoint file (required)
  --output DIR            Output directory (default: dashboard_output)
  --max-samples INT       Maximum molecules to include (default: 100)
```

---

## Project Structure

```
Inter_gnn/
├── pyproject.toml                              # Package configuration & dependencies
├── README.md                                   # This documentation
├── inter_gnn/                                  # Main package
│   ├── __init__.py                             # Package root: version, key exports
│   ├── cli.py                                  # Command-line interface
│   │
│   ├── data/                                   # ─── Data & Preprocessing ───
│   │   ├── __init__.py                         # Public API exports
│   │   ├── standardize.py                      # Molecule standardization pipeline
│   │   ├── featurize.py                        # SMILES → molecular graph features
│   │   ├── protein.py                          # Protein sequence → graph
│   │   ├── concepts.py                         # SMARTS concept library (~30 patterns)
│   │   ├── cliffs.py                           # Activity cliff detection
│   │   ├── splits.py                           # Scaffold, cold-target, temporal splits
│   │   ├── datasets.py                         # 9 benchmark dataset loaders
│   │   └── datamodule.py                       # DataModule wrapper
│   │
│   ├── models/                                 # ─── Core Model ───
│   │   ├── __init__.py
│   │   ├── encoders.py                         # MolecularGNNEncoder + TargetGNNEncoder
│   │   ├── attention.py                        # CrossAttentionFusion + BilinearFusion
│   │   ├── task_heads.py                       # ClassificationHead + RegressionHead
│   │   └── core_model.py                       # Unified InterGNN model
│   │
│   ├── interpretability/                       # ─── Intrinsic Interpretability ───
│   │   ├── __init__.py
│   │   ├── prototypes.py                       # PAGE prototype layer + losses
│   │   ├── motifs.py                           # MAGE motif generator + extractor
│   │   ├── concept_whitening.py                # ZCA whitening + concept alignment
│   │   └── stability.py                        # Explanation stability regularizer
│   │
│   ├── explainers/                             # ─── Post-hoc Explanations ───
│   │   ├── __init__.py
│   │   ├── cf_explainer.py                     # CF-GNNExplainer (counterfactual)
│   │   ├── t_explainer.py                      # T-GNNExplainer (sufficient subgraph)
│   │   └── cider.py                            # CIDER causal invariance diagnostics
│   │
│   ├── training/                               # ─── Training Pipeline ───
│   │   ├── __init__.py
│   │   ├── losses.py                           # Combined multi-objective loss (9 terms)
│   │   ├── trainer.py                          # Two-phase trainer
│   │   ├── callbacks.py                        # EarlyStopping, checkpointing, monitoring
│   │   └── config.py                           # YAML configuration dataclasses
│   │
│   ├── evaluation/                             # ─── Evaluation Metrics ───
│   │   ├── __init__.py
│   │   ├── predictive.py                       # ROC-AUC, PR-AUC, RMSE, CI, etc.
│   │   ├── faithfulness.py                     # Deletion/Insertion AUC
│   │   ├── stability_metrics.py                # Jaccard stability, cliff consistency
│   │   ├── chemical_validity.py                # Valence checks, SMARTS match rates
│   │   ├── causal.py                           # Invariance scores
│   │   └── statistical.py                      # Bootstrap & randomization tests
│   │
│   └── visualization/                          # ─── Visualization Tools ───
│       ├── __init__.py
│       ├── molecule_viz.py                     # Atom/bond saliency rendering
│       ├── prototype_viz.py                    # Prototype gallery & distances
│       ├── motif_viz.py                        # Motif activation heatmaps
│       ├── concept_viz.py                      # Concept activation bar charts
│       ├── counterfactual_viz.py               # Counterfactual edit comparison
│       └── dashboard.py                        # HTML batch-export dashboard
```

**Total: 44 Python files | ~6,500 lines of code**

---

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork** the repository and create a feature branch
2. **Install dev dependencies:** `pip install -e ".[dev]"`
3. **Code style:** Run `black inter_gnn/` and `ruff check inter_gnn/`
4. **Type checking:** Run `mypy inter_gnn/`
5. **Testing:** Add tests in `tests/` and run `pytest`
6. **Documentation:** Update this README for any new features

---

## Citation

If you use InterGNN in your research, please cite:

```bibtex
@software{inter_gnn2025,
  title     = {InterGNN: Interpretable Graph Neural Network Framework
               for Drug Discovery and Candidate Screening},
  year      = {2025},
  url       = {https://pypi.org/project/inter-gnn/},
  keywords  = {graph neural networks, drug discovery, explainable AI,
               molecular property prediction, interpretability},
}
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <strong>InterGNN</strong> — Making GNN predictions in drug discovery transparent, trustworthy, and scientifically actionable.
</p>
