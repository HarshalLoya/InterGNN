"""
Dataset loaders for all benchmark datasets.

Provides unified loading interface for MUTAG, Tox21, ClinTox, SIDER, 
BBBP, BACE, HIV, OGBG-MolHIV, BindingDB, and SynLethDB datasets.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset

from inter_gnn.data.featurize import smiles_to_graph
from inter_gnn.data.protein import ProteinGraphBuilder

logger = logging.getLogger(__name__)
DEFAULT_DATA_DIR = os.path.join(os.path.expanduser("~"), ".inter_gnn", "data")


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


class DTIData(Data):
    """Custom Data class for drug-target interaction data with dual graphs."""
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_target':
            return self.x_target.size(0)
        return super().__inc__(key, value, *args, **kwargs)


class InterGNNDataset(InMemoryDataset):
    """Base dataset wrapping a list of Data objects with metadata."""

    def __init__(self, data_list, dataset_name="unknown", task_type="classification",
                 num_tasks=1, smiles_list=None, cliff_pairs=None, concept_matrix=None):
        super().__init__(root=None)
        self._data_list = data_list
        self.dataset_name = dataset_name
        self.task_type = task_type
        self.num_tasks = num_tasks
        self.smiles_list = smiles_list or []
        self.cliff_pairs = cliff_pairs or []
        self.concept_matrix = concept_matrix

    def len(self):
        return len(self._data_list)

    def get(self, idx):
        data = self._data_list[idx]
        if self.concept_matrix is not None and idx < len(self.concept_matrix):
            data.concept_vector = torch.tensor(self.concept_matrix[idx], dtype=torch.float32)
        return data


def _load_pyg_moleculenet(name, root, num_tasks):
    from torch_geometric.datasets import MoleculeNet
    _ensure_dir(root)
    dataset = MoleculeNet(root=root, name=name)
    data_list, smiles_list = [], []
    for i in range(len(dataset)):
        d = dataset[i]; d.idx = i; data_list.append(d)
        if hasattr(d, "smiles"): smiles_list.append(d.smiles)
    logger.info(f"Loaded {name}: {len(data_list)} molecules")
    return InterGNNDataset(data_list, name, "classification", num_tasks, smiles_list)


def load_mutag(data_dir=None):
    """Load MUTAG (188 molecules, 2-class mutagenicity)."""
    from torch_geometric.datasets import TUDataset
    root = data_dir or os.path.join(DEFAULT_DATA_DIR, "MUTAG"); _ensure_dir(root)
    ds = TUDataset(root=root, name="MUTAG")
    data_list = [ds[i] for i in range(len(ds))]
    for i, d in enumerate(data_list): d.idx = i
    return InterGNNDataset(data_list, "MUTAG", "classification", 1)


def load_tox21(data_dir=None):
    """Load Tox21 (~8k compounds, 12 assays)."""
    return _load_pyg_moleculenet("Tox21", data_dir or os.path.join(DEFAULT_DATA_DIR, "Tox21"), 12)


def load_clintox(data_dir=None):
    """Load ClinTox (~1.5k compounds, 2 tasks)."""
    return _load_pyg_moleculenet("ClinTox", data_dir or os.path.join(DEFAULT_DATA_DIR, "ClinTox"), 2)


def load_bbbp(data_dir=None):
    """Load BBBP (Blood-Brain Barrier Permeability)."""
    return _load_pyg_moleculenet("BBBP", data_dir or os.path.join(DEFAULT_DATA_DIR, "BBBP"), 1)


def load_bace(data_dir=None):
    """Load BACE (Inhibitors of human beta-secretase 1)."""
    return _load_pyg_moleculenet("BACE", data_dir or os.path.join(DEFAULT_DATA_DIR, "BACE"), 1)


def load_hiv(data_dir=None):
    """Load HIV (Inhibition of HIV replication)."""
    return _load_pyg_moleculenet("HIV", data_dir or os.path.join(DEFAULT_DATA_DIR, "HIV"), 1)


def load_sider(data_dir=None):
    """Load SIDER (drug-ADR associations)."""
    return _load_pyg_moleculenet("SIDER", data_dir or os.path.join(DEFAULT_DATA_DIR, "SIDER"), 27)


def _build_dta_dataset(drug_smiles, target_seqs, affinities, name):
    """Helper to build DTA dataset from parallel lists."""
    builder = ProteinGraphBuilder(k=10)
    data_list, smiles_list, prot_cache = [], [], {}
    for idx, (smi, seq, aff) in enumerate(zip(drug_smiles, target_seqs, affinities)):
        dg = smiles_to_graph(smi, y=torch.tensor([aff], dtype=torch.float32))
        if dg is None: continue
        if seq not in prot_cache:
            pg = builder.from_sequence(seq)
            if pg is not None: prot_cache[seq] = pg
        pg = prot_cache.get(seq)
        if pg is None: continue
        data = DTIData(
            x=dg.x, edge_index=dg.edge_index, edge_attr=dg.edge_attr,
            x_target=pg.x, edge_index_target=pg.edge_index, edge_attr_target=pg.edge_attr,
            y=torch.tensor([aff], dtype=torch.float32), idx=idx, smiles=smi, sequence=seq,
        )
        data.num_nodes = dg.x.size(0)
        data.num_nodes_target = pg.x.size(0)
        data_list.append(data); smiles_list.append(smi)
    logger.info(f"Loaded {name}: {len(data_list)} pairs")
    return InterGNNDataset(data_list, name, "regression", 1, smiles_list)


def _load_tdc_dti(name, data_dir):
    """Load a DTI dataset via TDC or local CSV."""
    df = None
    try:
        from tdc.multi_pred import DTI
        df = DTI(name=name).get_data()
    except (ImportError, Exception):
        csv_file = os.path.join(data_dir or DEFAULT_DATA_DIR, name.lower(), f"{name.lower()}.csv")
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
        else:
            raise FileNotFoundError(f"{name} not found. Install PyTDC or place CSV at {csv_file}")

    # Robust column mapping for various naming conventions
    rename_dict = {
        "compound_iso_smiles": "Drug", "SMILES": "Drug", "smiles": "Drug",
        "target_sequence": "Target", "Protein": "Target", "target": "Target",
        "affinity": "Y", "label": "Y", "pkd": "Y", "value": "Y"
    }
    df = df.rename(columns=rename_dict)
    
    # Validation
    for col in ["Drug", "Target", "Y"]:
        if col not in df.columns:
            available = ", ".join(df.columns)
            raise KeyError(f"Missing required DTI column '{col}' in {name}. Available: {available}")

    return df["Drug"].tolist(), df["Target"].tolist(), df["Y"].tolist()


def load_ogbg_molhiv(data_dir=None):
    """Load OGBG-MolHIV dataset via ogb.graphproppred."""
    try:
        from ogb.graphproppred import PygGraphPropPredDataset
    except ImportError:
        logger.warning("ogb not installed. Falling back to MoleculeNet HIV.")
        return load_hiv(data_dir)
        
    root = data_dir or os.path.join(DEFAULT_DATA_DIR, "ogbg_molhiv"); _ensure_dir(root)
    dataset = PygGraphPropPredDataset(name="ogbg-molhiv", root=root)
    data_list, smiles_list = [], []
    # OGB datasets don't give smiles natively via data object but we can just use the graph objects
    for i in range(len(dataset)):
        d = dataset[i]
        d.idx = i
        # Convert y to float32 for BCEWithLogitsLoss
        d.y = d.y.to(torch.float32)
        data_list.append(d)
        
    logger.info(f"Loaded ogbg-molhiv: {len(data_list)} molecules")
    return InterGNNDataset(data_list, "ogbg_molhiv", "classification", 1)


def load_bindingdb(data_dir=None, affinity_type="Kd", max_records=None):
    """Load BindingDB (protein-small molecule binding data)."""
    try:
        from tdc.multi_pred import DTI
        df = DTI(name="BindingDB_" + affinity_type).get_data()
    except ImportError:
        csv = os.path.join(data_dir or DEFAULT_DATA_DIR, "bindingdb", "bindingdb.csv")
        if os.path.exists(csv): df = pd.read_csv(csv)
        else: raise FileNotFoundError(f"BindingDB not found at {csv}")
    if max_records: df = df.head(max_records)
    return _build_dta_dataset(df["Drug"].tolist(), df["Target"].tolist(), df["Y"].tolist(), "BindingDB")


def load_synlethdb(data_dir=None):
    """Load SynLethDB (synthetic lethal gene pairs)."""
    try:
        from tdc.multi_pred import GDA
        df = GDA(name="SynLethDB").get_data()
    except ImportError:
        csv = os.path.join(data_dir or DEFAULT_DATA_DIR, "synlethdb", "synlethdb.csv")
        if os.path.exists(csv): df = pd.read_csv(csv)
        else: raise FileNotFoundError("SynLethDB not found.")
    genes = set()
    if "Gene1" in df.columns: genes = set(df["Gene1"].tolist() + df["Gene2"].tolist())
    gene_idx = {g: i for i, g in enumerate(sorted(genes))}
    if genes:
        src = [gene_idx[g] for g in df["Gene1"]]; dst = [gene_idx[g] for g in df["Gene2"]]
        ei = torch.tensor([src + dst, dst + src], dtype=torch.long)
        labels = df["Y"].values if "Y" in df.columns else np.ones(len(df))
        data_list = [Data(num_nodes=len(gene_idx), edge_index=ei,
                          y=torch.tensor(labels, dtype=torch.float32))]
    else:
        data_list = []
    return InterGNNDataset(data_list, "SynLethDB", "link_prediction", 1)


DATASET_REGISTRY = {
    "mutag": load_mutag, "tox21": load_tox21, "clintox": load_clintox,
    "sider": load_sider, "bbbp": load_bbbp, "bace": load_bace,
    "hiv": load_hiv, "ogbg_molhiv": load_ogbg_molhiv,
    "bindingdb": load_bindingdb, "synlethdb": load_synlethdb,
}


def list_datasets() -> List[str]:
    """Return names of all available datasets."""
    return sorted(DATASET_REGISTRY.keys())


def load_dataset(name: str, data_dir=None, **kwargs) -> InterGNNDataset:
    """Unified loader for classification benchmarks."""
    key = name.lower()
    if key not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Available: {sorted(DATASET_REGISTRY)}")
    return DATASET_REGISTRY[key](data_dir=data_dir, **kwargs)

