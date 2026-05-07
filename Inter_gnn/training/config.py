"""
YAML configuration loader and hyperparameter management.

Provides hierarchical configuration for all InterGNN components
with sensible defaults and YAML override support.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    dataset_name: str = "mutag"
    data_dir: Optional[str] = None
    split_method: str = "scaffold"
    frac_train: float = 0.8
    frac_val: float = 0.1
    frac_test: float = 0.1
    batch_size: int = 64
    num_workers: int = 0
    detect_cliffs: bool = False
    cliff_sim_threshold: float = 0.9
    cliff_act_threshold: float = 1.0
    compute_concepts: bool = False
    seed: int = 42


@dataclass
class ModelConfig:
    atom_feat_dim: int = 55
    bond_feat_dim: int = 14
    hidden_dim: int = 256
    num_mol_layers: int = 4
    task_type: str = "classification"
    num_tasks: int = 1
    dropout: float = 0.2
    readout: str = "attention"


@dataclass
class InterpretabilityConfig:
    use_prototypes: bool = False
    num_prototypes_per_class: int = 5
    prototype_activation: str = "log"

    use_motifs: bool = False
    num_motifs: int = 8
    motif_temperature: float = 0.5
    motif_sparsity_target: float = 0.3

    use_concept_whitening: bool = False
    num_concepts: int = 30
    concept_momentum: float = 0.1

    use_stability: bool = False
    stability_mask_prob: float = 0.1
    stability_edge_drop_prob: float = 0.05


@dataclass
class LossConfig:
    lambda_pred: float = 1.0
    lambda_pull: float = 0.1
    lambda_push: float = 0.05
    lambda_diversity: float = 0.01
    lambda_motif: float = 0.1
    lambda_connectivity: float = 0.05
    lambda_concept: float = 0.1
    lambda_decorrelation: float = 0.01
    lambda_stability: float = 0.05


@dataclass
class TrainingConfig:
    pretrain_epochs: int = 30
    finetune_epochs: int = 30
    learning_rate: float = 1e-2
    weight_decay: float = 1e-5
    lr_scheduler: str = "cosine"
    warmup_steps: int = 500
    gradient_clip: float = 1.0
    early_stopping_patience: int = 15
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 10
    eval_interval: int = 1
    device: str = "auto"
    seed: int = 42


@dataclass
class InterGNNConfig:
    """
    Master configuration for the entire InterGNN pipeline.

    Can be loaded from YAML files and provides sensible defaults
    for all sub-configurations.

    Example YAML::

        data:
          dataset_name: tox21
          split_method: scaffold
          batch_size: 32

        model:
          hidden_dim: 256
          num_mol_layers: 4
          task_type: classification

        interpretability:
          use_prototypes: true
          num_prototypes_per_class: 5

        training:
          pretrain_epochs: 50
          finetune_epochs: 100
    """

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    interpretability: InterpretabilityConfig = field(default_factory=InterpretabilityConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_yaml(cls, path: str) -> InterGNNConfig:
        """Load config from YAML file."""
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}

        config = cls()
        if "data" in raw:
            for k, v in raw["data"].items():
                if hasattr(config.data, k):
                    setattr(config.data, k, v)
        if "model" in raw:
            for k, v in raw["model"].items():
                if hasattr(config.model, k):
                    setattr(config.model, k, v)
        if "interpretability" in raw:
            for k, v in raw["interpretability"].items():
                if hasattr(config.interpretability, k):
                    setattr(config.interpretability, k, v)
        if "loss" in raw:
            for k, v in raw["loss"].items():
                if hasattr(config.loss, k):
                    setattr(config.loss, k, v)
        if "training" in raw:
            for k, v in raw["training"].items():
                if hasattr(config.training, k):
                    setattr(config.training, k, v)

        logger.info(f"Config loaded from {path}")
        return config

    def to_yaml(self, path: str):
        """Save config to YAML file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)
        logger.info(f"Config saved to {path}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
