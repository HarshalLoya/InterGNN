"""
Two-phase training loop for InterGNN.

Phase 1 (Pre-training): Train encoders + task head only with prediction loss.
Phase 2 (Joint Fine-tuning): Joint training with all interpretability losses.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from tqdm import tqdm

from inter_gnn.models.core_model import InterGNN
from inter_gnn.training.losses import TotalLoss
from inter_gnn.training.config import InterGNNConfig
from inter_gnn.training.callbacks import CallbackManager, EarlyStopping, ModelCheckpoint
from inter_gnn.interpretability.prototypes import PrototypeLayer
from inter_gnn.interpretability.motifs import MotifGeneratorHead
from inter_gnn.interpretability.concept_whitening import ConceptWhiteningLayer
from inter_gnn.interpretability.stability import ExplanationStabilityLoss

logger = logging.getLogger(__name__)


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


class InterGNNTrainer:
    """
    Two-phase trainer for InterGNN.

    Phase 1 (Pre-training):
      - Trains molecular/target encoders + task head
      - Uses only prediction loss
      - Establishes baseline representation quality

    Phase 2 (Joint Fine-tuning):
      - Attaches interpretability modules (prototypes, motifs, concept whitening)
      - Trains all components jointly with combined loss
      - Adds stability regularization

    Example::

        config = InterGNNConfig.from_yaml("config.yaml")
        trainer = InterGNNTrainer(config)
        trainer.fit(train_loader, val_loader)
    """

    def __init__(self, config: InterGNNConfig):
        self.config = config
        self.device = _resolve_device(config.training.device)
        logger.info(f"InterGNNTrainer initialized on device: {self.device}")

        # Build model
        self.model = InterGNN(
            atom_feat_dim=config.model.atom_feat_dim,
            bond_feat_dim=config.model.bond_feat_dim,
            residue_feat_dim=config.model.residue_feat_dim,
            hidden_dim=config.model.hidden_dim,
            num_mol_layers=config.model.num_mol_layers,
            num_target_layers=config.model.num_target_layers,
            num_attn_heads=config.model.num_attn_heads,
            task_type=config.model.task_type,
            num_tasks=config.model.num_tasks,
            dropout=config.model.dropout,
            use_target=config.model.use_target,
            fusion_type=config.model.fusion_type,
            readout=config.model.readout,
        ).to(self.device)

        # Build loss
        self.loss_fn = TotalLoss(
            task_type=config.model.task_type,
            num_tasks=config.model.num_tasks,
            lambda_pred=config.loss.lambda_pred,
            lambda_pull=config.loss.lambda_pull,
            lambda_push=config.loss.lambda_push,
            lambda_diversity=config.loss.lambda_diversity,
            lambda_motif=config.loss.lambda_motif,
            lambda_connectivity=config.loss.lambda_connectivity,
            lambda_concept=config.loss.lambda_concept,
            lambda_decorrelation=config.loss.lambda_decorrelation,
            lambda_stability=config.loss.lambda_stability,
        )

        self.history: List[Dict] = []

    def _build_optimizer(self, lr: float) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=self.config.training.weight_decay,
        )

    def _build_scheduler(self, optimizer, num_epochs):
        sched = self.config.training.lr_scheduler
        if sched == "cosine":
            return CosineAnnealingLR(optimizer, T_max=num_epochs)
        elif sched == "step":
            return StepLR(optimizer, step_size=num_epochs // 3, gamma=0.1)
        elif sched == "plateau":
            return ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        return None

    def _attach_interpretability(self):
        """Attach interpretability modules for Phase 2."""
        cfg = self.config.interpretability

        if cfg.use_prototypes:
            # For classification, prototypes are always binary (active/inactive).
            # Multi-label datasets (num_tasks > 1) use a derived pseudo-label.
            num_classes = 2 if self.config.model.task_type == "classification" else max(self.config.model.num_tasks, 2)
            self.model.prototype_layer = PrototypeLayer(
                hidden_dim=self.config.model.hidden_dim,
                num_classes=num_classes,
                num_prototypes_per_class=cfg.num_prototypes_per_class,
                prototype_activation=cfg.prototype_activation,
            ).to(self.device)

        if cfg.use_motifs:
            self.model.motif_head = MotifGeneratorHead(
                hidden_dim=self.config.model.hidden_dim,
                num_motifs=cfg.num_motifs,
                temperature=cfg.motif_temperature,
                sparsity_target=cfg.motif_sparsity_target,
            ).to(self.device)

        if cfg.use_concept_whitening:
            self.model.concept_whitening = ConceptWhiteningLayer(
                hidden_dim=self.config.model.hidden_dim,
                num_concepts=cfg.num_concepts,
                momentum=cfg.concept_momentum,
            ).to(self.device)

    def _train_epoch(self, loader, optimizer, phase="pretrain", accumulation_steps=1) -> Dict:
        self.model.train()
        total_loss = 0.0
        loss_components = {}
        num_batches = 0
        
        # Determine actual accumulation steps (only accumulate if steps > 1)
        # Avoid dividing by 0
        acc_steps = max(1, accumulation_steps)
        optimizer.zero_grad(set_to_none=True)

        for batch_data in tqdm(loader, desc=f"Training ({phase})", leave=False, disable=(len(loader) < 10)):
            batch_data = batch_data.to(self.device)

            # Build forward kwargs
            kwargs = {
                "x": batch_data.x,
                "edge_index": batch_data.edge_index,
                "edge_attr": batch_data.edge_attr,
                "batch": batch_data.batch,
            }

            # DTA mode
            if hasattr(batch_data, "x_target") and batch_data.x_target is not None:
                kwargs["x_target"] = batch_data.x_target
                kwargs["edge_index_target"] = batch_data.edge_index_target
                kwargs["edge_attr_target"] = batch_data.edge_attr_target
                # If using follow_batch=['x_target'], PyG creates x_target_batch
                kwargs["batch_target"] = getattr(batch_data, "x_target_batch", 
                                                 getattr(batch_data, "batch_target", None))

            # Concept labels
            if hasattr(batch_data, "concept_vector"):
                kwargs["concept_labels"] = batch_data.concept_vector

            output = self.model(**kwargs)

            # Compute loss
            losses = self.loss_fn(
                output, batch_data.y,
                model=self.model if phase == "finetune" else None,
                edge_index=batch_data.edge_index,
                batch=batch_data.batch,
            )

            # Scale loss for gradient accumulation
            scaled_loss = losses["total"] / acc_steps
            scaled_loss.backward()
            
            # Step optimizer every `acc_steps` batches or on the last batch
            if (num_batches + 1) % acc_steps == 0 or (num_batches + 1) == len(loader):
                if self.config.training.gradient_clip > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.training.gradient_clip
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            total_loss += losses["total"].item()
            for k, v in losses.items():
                if k not in loss_components:
                    loss_components[k] = 0.0
                loss_components[k] += v.item() if isinstance(v, torch.Tensor) else v
            num_batches += 1

        avg_losses = {k: v / max(num_batches, 1) for k, v in loss_components.items()}
        avg_losses["epoch_total"] = total_loss / max(num_batches, 1)
        return avg_losses

    @torch.inference_mode()
    def _eval_epoch(self, loader) -> Dict:
        self.model.eval()
        total_loss = 0.0
        pred_loss_total = 0.0
        all_preds, all_targets = [], []
        num_batches = 0

        for batch_data in loader:
            batch_data = batch_data.to(self.device)
            kwargs = {
                "x": batch_data.x,
                "edge_index": batch_data.edge_index,
                "edge_attr": batch_data.edge_attr,
                "batch": batch_data.batch,
            }
            if hasattr(batch_data, "x_target") and batch_data.x_target is not None:
                kwargs["x_target"] = batch_data.x_target
                kwargs["edge_index_target"] = batch_data.edge_index_target
                kwargs["edge_attr_target"] = batch_data.edge_attr_target
                kwargs["batch_target"] = getattr(batch_data, "x_target_batch", 
                                                 getattr(batch_data, "batch_target", None))

            output = self.model(**kwargs)
            # Loss is computed on raw logits (head now always returns logits)
            losses = self.loss_fn(output, batch_data.y)
            total_loss += losses["total"].item()
            # Track prediction loss separately for early stopping / checkpointing
            pred_loss_total += losses["prediction"].item()

            # Apply sigmoid/softmax for metric-ready predictions
            pred = output["prediction"]
            if self.config.model.task_type == "classification":
                pred = torch.sigmoid(pred)

            all_preds.append(pred.cpu())
            all_targets.append(batch_data.y.cpu())
            num_batches += 1

        return {
            "val_loss": pred_loss_total / max(num_batches, 1),
            "val_total_loss": total_loss / max(num_batches, 1),
            "predictions": torch.cat(all_preds, dim=0),
            "targets": torch.cat(all_targets, dim=0),
        }

    def fit(self, train_loader, val_loader=None):
        """
        Run the full two-phase training pipeline.

        Phase 1: Pre-train encoders + task head.
        Phase 2: Attach interpretability modules and fine-tune jointly.
        """
        os.makedirs(self.config.training.checkpoint_dir, exist_ok=True)

        # ── Phase 1: Pre-training ──
        logger.info("=== Phase 1: Pre-training (prediction only) ===")
        optimizer = self._build_optimizer(self.config.training.learning_rate)
        scheduler = self._build_scheduler(optimizer, self.config.training.pretrain_epochs)
        early_stop = EarlyStopping(patience=self.config.training.early_stopping_patience)
        ckpt = ModelCheckpoint(self.config.training.checkpoint_dir, "pretrain")
        
        # Accumulate gradients specifically for the Davis dataset
        # We accumulate 2 steps with batch size 8 = effective batch size 16
        is_davis = getattr(self.config.data, "dataset_name", "") == "davis"
        acc_steps = 2 if is_davis else 1

        for epoch in range(1, self.config.training.pretrain_epochs + 1):
            train_metrics = self._train_epoch(train_loader, optimizer, "pretrain", accumulation_steps=acc_steps)
            val_metrics = self._eval_epoch(val_loader) if val_loader else {}

            val_loss = val_metrics.get("val_loss", train_metrics["epoch_total"])
            if scheduler and not isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step()
            elif scheduler:
                scheduler.step(val_loss)

            ckpt.step(val_loss, self.model, epoch)
            if early_stop.step(val_loss):
                logger.info(f"Early stopping at epoch {epoch}")
                break

            if epoch % self.config.training.log_interval == 0:
                logger.info(
                    f"[Pretrain] Epoch {epoch}: "
                    f"train_loss={train_metrics['epoch_total']:.4f}, "
                    f"val_loss={val_loss:.4f}"
                )

            self.history.append({
                "phase": "pretrain", "epoch": epoch,
                **train_metrics,
                "val_loss": val_metrics.get("val_loss", train_metrics["epoch_total"]),
            })

        # ── Phase 2: Joint Fine-tuning ──
        logger.info("=== Phase 2: Joint fine-tuning (all losses) ===")
        self._attach_interpretability()

        optimizer = self._build_optimizer(self.config.training.learning_rate * 0.1)
        scheduler = self._build_scheduler(optimizer, self.config.training.finetune_epochs)
        early_stop = EarlyStopping(patience=self.config.training.early_stopping_patience)
        ckpt = ModelCheckpoint(self.config.training.checkpoint_dir, "finetune")
        
        is_davis = getattr(self.config.data, "dataset_name", "") == "davis"
        acc_steps = 2 if is_davis else 1

        for epoch in range(1, self.config.training.finetune_epochs + 1):
            train_metrics = self._train_epoch(train_loader, optimizer, "finetune", accumulation_steps=acc_steps)
            val_metrics = self._eval_epoch(val_loader) if val_loader else {}

            val_loss = val_metrics.get("val_loss", train_metrics["epoch_total"])
            if scheduler and not isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step()
            elif scheduler:
                scheduler.step(val_loss)

            ckpt.step(val_loss, self.model, epoch)
            if early_stop.step(val_loss):
                logger.info(f"Early stopping at epoch {epoch}")
                break

            if epoch % self.config.training.log_interval == 0:
                logger.info(
                    f"[Finetune] Epoch {epoch}: "
                    f"train_loss={train_metrics['epoch_total']:.4f}, "
                    f"val_loss={val_loss:.4f}"
                )

            self.history.append({
                "phase": "finetune", "epoch": epoch,
                **train_metrics,
                "val_loss": val_metrics.get("val_loss", train_metrics["epoch_total"]),
            })

        logger.info("Training complete.")
        return self.history
