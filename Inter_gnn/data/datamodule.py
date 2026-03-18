"""
DataModule wrapper for InterGNN experiments.

Provides a unified interface for dataset loading, splitting, cliff-pair
attachment, concept vector computation, and DataLoader creation.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
from torch_geometric.loader import DataLoader

from inter_gnn.data.datasets import InterGNNDataset, load_dataset
from inter_gnn.data.splits import scaffold_split, cold_target_split, temporal_split, random_split
from inter_gnn.data.cliffs import find_cliff_pairs, get_cliff_pair_indices
from inter_gnn.data.concepts import ConceptDataset

logger = logging.getLogger(__name__)


class InterGNNDataModule:
    """
    Manages data loading, splitting, and batching for InterGNN training.

    Example::

        dm = InterGNNDataModule(
            dataset_name="tox21",
            split_method="scaffold",
            batch_size=64,
            detect_cliffs=True,
        )
        dm.setup()
        train_loader = dm.train_dataloader()
    """

    def __init__(
        self,
        dataset_name: str = "mutag",
        data_dir: Optional[str] = None,
        split_method: str = "scaffold",
        frac_train: float = 0.8,
        frac_val: float = 0.1,
        frac_test: float = 0.1,
        batch_size: int = 64,
        num_workers: int = 0,
        seed: int = 42,
        detect_cliffs: bool = False,
        cliff_sim_threshold: float = 0.9,
        cliff_act_threshold: float = 1.0,
        compute_concepts: bool = False,
        **dataset_kwargs,
    ):
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.split_method = split_method
        self.frac_train = frac_train
        self.frac_val = frac_val
        self.frac_test = frac_test
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.detect_cliffs = detect_cliffs
        self.cliff_sim_threshold = cliff_sim_threshold
        self.cliff_act_threshold = cliff_act_threshold
        self.compute_concepts = compute_concepts
        self.dataset_kwargs = dataset_kwargs

        self.dataset: Optional[InterGNNDataset] = None
        self.split_indices: Optional[Dict[str, List[int]]] = None
        self.cliff_pairs: List = []
        self.concept_dataset: Optional[ConceptDataset] = None

    def setup(self):
        """Load dataset, compute splits, detect cliffs, and build concept vectors."""
        # Load dataset
        self.dataset = load_dataset(self.dataset_name, self.data_dir, **self.dataset_kwargs)
        logger.info(f"Dataset loaded: {self.dataset.dataset_name} ({self.dataset.len()} samples)")

        # Compute split
        n = self.dataset.len()
        if self.split_method == "scaffold" and self.dataset.smiles_list:
            self.split_indices = scaffold_split(
                self.dataset.smiles_list, self.frac_train, self.frac_val, self.frac_test,
                seed=self.seed,
            )
        elif self.split_method == "random":
            self.split_indices = random_split(
                n, self.frac_train, self.frac_val, self.frac_test, seed=self.seed,
            )
        else:
            self.split_indices = random_split(
                n, self.frac_train, self.frac_val, self.frac_test, seed=self.seed,
            )

        # Detect activity cliffs (training set only)
        if self.detect_cliffs and self.dataset.smiles_list:
            train_smiles = [self.dataset.smiles_list[i] for i in self.split_indices["train"]]
            train_activities = []
            for i in self.split_indices["train"]:
                d = self.dataset.get(i)
                if hasattr(d, "y") and d.y is not None:
                    train_activities.append(float(d.y.flatten()[0]))
                else:
                    train_activities.append(0.0)
            if train_activities:
                self.cliff_pairs = find_cliff_pairs(
                    train_smiles, train_activities,
                    sim_threshold=self.cliff_sim_threshold,
                    act_threshold=self.cliff_act_threshold,
                )
                self.dataset.cliff_pairs = get_cliff_pair_indices(self.cliff_pairs)

        # Build concept vectors
        if self.compute_concepts and self.dataset.smiles_list:
            self.concept_dataset = ConceptDataset(self.dataset.smiles_list)
            self.concept_dataset.build_concept_examples()
            self.dataset.concept_matrix = self.concept_dataset.concept_matrix

    def _get_subset(self, split: str) -> List:
        indices = self.split_indices[split]
        return [self.dataset.get(i) for i in indices]

    def _dataloader_kwargs(self) -> dict:
        """Returns kwargs for DataLoader initialization."""
        kwargs = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
        }
        # For DTI datasets, we need to generate batch indicator for the target protein graph
        if self.dataset_name.lower() in ["davis", "kiba", "bindingdb"]:
            kwargs["follow_batch"] = ["x_target"]
        return kwargs

    def train_dataloader(self) -> DataLoader:
        subset = self._get_subset("train")
        return DataLoader(
            subset, shuffle=True,
            drop_last=(len(subset) > self.batch_size * 2),
            **self._dataloader_kwargs()
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._get_subset("val"), shuffle=False,
            **self._dataloader_kwargs()
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._get_subset("test"), shuffle=False,
            **self._dataloader_kwargs()
        )
