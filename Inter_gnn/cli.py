"""
Command-line interface for InterGNN.

Usage:
    inter-gnn train    --config config.yaml
    inter-gnn evaluate --config config.yaml --checkpoint model.pt
    inter-gnn explain  --config config.yaml --checkpoint model.pt --smiles "CCO"
    inter-gnn dashboard --config config.yaml --checkpoint model.pt --output report/
"""

from __future__ import annotations

import argparse
import logging
import sys

import torch

from inter_gnn.training.config import InterGNNConfig

logger = logging.getLogger("inter_gnn")


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def cmd_train(args):
    """Run the two-phase training pipeline."""
    from inter_gnn.training.trainer import InterGNNTrainer
    from inter_gnn.data.datamodule import InterGNNDataModule

    config = InterGNNConfig.from_yaml(args.config)
    logger.info(f"Training with config: {args.config}")

    # Build data module
    dm = InterGNNDataModule(config)
    dm.prepare_data()
    dm.setup()

    # Train
    trainer = InterGNNTrainer(config)
    history = trainer.fit(dm.train_dataloader(), dm.val_dataloader())

    logger.info(f"Training complete. {len(history)} epochs recorded.")
    if args.save_config:
        config.to_yaml(args.save_config)


def cmd_evaluate(args):
    """Evaluate a trained model."""
    from inter_gnn.training.trainer import InterGNNTrainer
    from inter_gnn.data.datamodule import InterGNNDataModule
    from inter_gnn.evaluation.predictive import (
        compute_classification_metrics,
        compute_regression_metrics,
    )

    config = InterGNNConfig.from_yaml(args.config)
    dm = InterGNNDataModule(config)
    dm.prepare_data()
    dm.setup()

    trainer = InterGNNTrainer(config)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=trainer.device)
    trainer.model.load_state_dict(ckpt["model_state_dict"])
    logger.info(f"Loaded checkpoint: {args.checkpoint}")

    # Evaluate
    results = trainer._eval_epoch(dm.test_dataloader())
    preds = results["predictions"].numpy()
    targets = results["targets"].numpy()

    if config.model.task_type == "classification":
        metrics = compute_classification_metrics(preds, targets)
    else:
        metrics = compute_regression_metrics(preds, targets)

    for name, value in metrics.items():
        logger.info(f" {name}: {value:.4f}")

    return metrics


def cmd_explain(args):
    """Generate explanations for molecules."""
    from inter_gnn.training.trainer import InterGNNTrainer
    from inter_gnn.data.featurize import smiles_to_graph
    import json

    config = InterGNNConfig.from_yaml(args.config)
    trainer = InterGNNTrainer(config)

    ckpt = torch.load(args.checkpoint, map_location=trainer.device)
    trainer.model.load_state_dict(ckpt["model_state_dict"])
    trainer.model.eval()

    smiles_list = args.smiles if isinstance(args.smiles, list) else [args.smiles]
    results = []

    for smi in smiles_list:
        graph = smiles_to_graph(smi)
        if graph is None:
            logger.warning(f"Invalid SMILES: {smi}")
            continue

        graph = graph.to(trainer.device)
        batch = torch.zeros(graph.x.shape[0], dtype=torch.long, device=trainer.device)

        with torch.no_grad():
            output = trainer.model(graph.x, graph.edge_index, graph.edge_attr, batch)

        importance = trainer.model.get_node_importance(
            graph.x, graph.edge_index, graph.edge_attr, batch
        )

        result = {
            "smiles": smi,
            "prediction": output["prediction"].cpu().tolist(),
            "atom_importance": importance.cpu().tolist(),
        }
        results.append(result)
        logger.info(f"Explained: {smi} → pred={output['prediction'].cpu().tolist()}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Explanations saved to {args.output}")

    return results


def cmd_dashboard(args):
    """Generate an explanation dashboard."""
    import json
    import numpy as np
    from inter_gnn.visualization.dashboard import ExplanationDashboard
    from inter_gnn.training.trainer import InterGNNTrainer
    from inter_gnn.data.datamodule import InterGNNDataModule

    config = InterGNNConfig.from_yaml(args.config)
    dm = InterGNNDataModule(config)
    dm.prepare_data()
    dm.setup()

    trainer = InterGNNTrainer(config)
    ckpt = torch.load(args.checkpoint, map_location=trainer.device)
    trainer.model.load_state_dict(ckpt["model_state_dict"])
    trainer.model.eval()

    dashboard = ExplanationDashboard(args.output, title=f"InterGNN — {config.data.dataset_name}")

    test_loader = dm.test_dataloader()
    count = 0

    for batch_data in test_loader:
        batch_data = batch_data.to(trainer.device)

        with torch.no_grad():
            output = trainer.model(
                batch_data.x, batch_data.edge_index, batch_data.edge_attr, batch_data.batch
            )

        importance = trainer.model.get_node_importance(
            batch_data.x, batch_data.edge_index, batch_data.edge_attr, batch_data.batch
        )

        preds = output["prediction"].cpu().numpy()
        targets = batch_data.y.cpu().numpy()

        # Add per-graph entries
        batch_np = batch_data.batch.cpu().numpy()
        for b in range(int(batch_np.max()) + 1):
            mask = batch_np == b
            smi = getattr(batch_data, "smiles", [f"mol_{count}"])[b] if hasattr(batch_data, "smiles") else f"mol_{count}"

            dashboard.add_entry(
                smiles=smi,
                prediction=float(preds[b].mean()),
                target=float(targets[b]) if targets.ndim == 1 else float(targets[b].mean()),
                atom_importance=importance[mask].cpu().numpy(),
            )
            count += 1

        if count >= args.max_samples:
            break

    path = dashboard.generate()
    logger.info(f"Dashboard saved: {path}")


def main():
    parser = argparse.ArgumentParser(
        prog="inter-gnn",
        description="InterGNN: Interpretable GNN for Drug Discovery",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── Train ──
    train_parser = subparsers.add_parser("train", help="Train InterGNN model")
    train_parser.add_argument("--config", required=True, help="YAML config file")
    train_parser.add_argument("--save-config", default=None, help="Save resolved config")

    # ── Evaluate ──
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate trained model")
    eval_parser.add_argument("--config", required=True)
    eval_parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")

    # ── Explain ──
    explain_parser = subparsers.add_parser("explain", help="Generate explanations")
    explain_parser.add_argument("--config", required=True)
    explain_parser.add_argument("--checkpoint", required=True)
    explain_parser.add_argument("--smiles", nargs="+", required=True, help="SMILES to explain")
    explain_parser.add_argument("--output", default=None, help="Output JSON path")

    # ── Dashboard ──
    dash_parser = subparsers.add_parser("dashboard", help="Generate explanation dashboard")
    dash_parser.add_argument("--config", required=True)
    dash_parser.add_argument("--checkpoint", required=True)
    dash_parser.add_argument("--output", default="dashboard_output", help="Output directory")
    dash_parser.add_argument("--max-samples", type=int, default=100)

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.command == "train":
        cmd_train(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "explain":
        cmd_explain(args)
    elif args.command == "dashboard":
        cmd_dashboard(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
