# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# Modified from solo-learn to meet the requirements of RepreSent

import os
import torch
import wandb
import torchmetrics

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from typing import Any, Dict, List, Sequence
from argparse import ArgumentParser
from datetime import datetime
from pytorch_lightning import seed_everything
from pl_bolts.models.self_supervised import resnets
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torchmetrics.classification import BinaryJaccardIndex, BinaryAccuracy


from represent.losses.functional import deepclusterv2_loss_func
from represent.tools.utils import remove_bias_and_norm_from_weight_decay
from represent.tools.metrics import accuracy_at_k, weighted_mean
from represent.tools.lars import LARS
from represent.tools.kmeans import KMeans


class DeepClusterV2(pl.LightningModule):
    def __init__(
        self,
        backbone: str = "resnet18",
        input_ch: int = 13,
        num_classes: int = 10,
        proj_hidden_dim: int = 2048,
        proj_output_dim: int = 128,
        num_prototypes: Sequence[int] = [3000, 3000, 3000],
        temperature: float = 0.1,
        kmeans_iters: int = 10,
        *args,
        **kwargs,
    ):
        """Implements DeepCluster V2 (https://arxiv.org/abs/2006.09882).

        Extra cfg settings:
            method_kwargs:
                proj_output_dim (int): number of dimensions of the projected features.
                proj_hidden_dim (int): number of neurons in the hidden layers of the projector.
                num_prototypes (Sequence[int]): number of prototypes.
                temperature (float): temperature for the softmax.
                kmeans_iters (int): number of iterations for k-means clustering.
        """
        super().__init__()
        self.save_hyperparameters()

        self.proj_output_dim: int = proj_output_dim
        self.temperature: float = temperature
        self.num_prototypes: Sequence[int] = num_prototypes
        self.kmeans_iters: int = kmeans_iters
        self.num_large_crops: int = 2
        self.num_classes = num_classes
        self.warmup_epochs = 11

        # Create the base model
        template_model = getattr(resnets, backbone)
        self.backbone = template_model(num_classes=2, return_all_feature_maps=False)

        self.backbone.conv1 = nn.Conv2d(input_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Initialize this correctly
        nn.init.kaiming_normal_(self.backbone.conv1.weight, mode="fan_out", nonlinearity="relu")

        dim_mlp = self.backbone.fc.weight.shape[1]
        self.backbone.fc = nn.Identity()

        self.features_dim: int = self.backbone.inplanes  # Not sure if this will work
        self.classifier = nn.Linear(self.features_dim, self.num_classes)

        self.proj_hidden_dim: int = proj_hidden_dim
        self.proj_output_dim: int = proj_output_dim

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, self.proj_hidden_dim),
            nn.BatchNorm1d(self.proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.proj_hidden_dim, self.proj_output_dim),
        )

        # prototypes
        self.prototypes = nn.ModuleList(
            [nn.Linear(self.proj_output_dim, np, bias=False) for np in self.num_prototypes]
        )

        # normalize and set requires grad to false
        for proto in self.prototypes:
            for params in proto.parameters():
                params.requires_grad = False
            proto.weight.copy_(F.normalize(proto.weight.data.clone(), dim=-1))

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--backbone", type=str, default="resnet18")
        parser.add_argument("--num_classes", type=int, default=21)
        parser.add_argument("--input_ch", type=int, default=13)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)
        parser.add_argument("--proj_output_dim", type=int, default=128)
        parser.add_argument("--temperature", type=float, default=0.1)
        parser.add_argument("--learning_rate", type=float, default=0.6)
        parser.add_argument("--classifier_lr", type=float, default=0.1)
        parser.add_argument("--weight_decay", type=float, default=1e-6)
        return parser

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector and prototypes parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        learnable_params = [
            {"name": "backbone", "params": self.backbone.parameters()},
            {
                "name": "classifier",
                "params": self.classifier.parameters(),
                "lr": self.hparams.classifier_lr,
                "weight_decay": 0,
            },
            {"name": "projector", "params": self.projector.parameters()},
        ]

        return learnable_params

    def configure_optimizers(self):
        learnable_params = remove_bias_and_norm_from_weight_decay(self.learnable_params)

        # create optimizer
        optimizer = LARS(
            learnable_params,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        max_warmup_steps = self.warmup_epochs * (self.trainer.estimated_stepping_batches / self.trainer.max_epochs)
        max_scheduler_steps = self.trainer.estimated_stepping_batches

        scheduler = {
            "scheduler": LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=max_warmup_steps,
                max_epochs=max_scheduler_steps,
                warmup_start_lr=0.0,
                eta_min=0.0006,
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]

    def on_train_start(self):
        """Gets the world size and initializes the memory banks."""
        #  k-means needs the world size and the dataset size
        self.world_size = self.trainer.world_size if self.trainer else 1

        try:
            self.dataset_size = len(self.trainer.train_dataloader.dataset)
        except:
            # get dataset size from dali
            self.dataset_size = self.trainer.train_dataloader.loaders.dataset_size

        # build k-means helper object
        self.kmeans = KMeans(
            world_size=self.world_size,
            rank=self.global_rank,
            num_large_crops=self.num_large_crops,
            dataset_size=self.dataset_size,
            proj_features_dim=self.proj_output_dim,
            num_prototypes=self.num_prototypes,
            kmeans_iters=self.kmeans_iters,
        )

        # initialize memory banks
        size_memory_per_process = len(self.trainer.train_dataloader) * self.hparams.batch_size
        self.register_buffer(
            "local_memory_index",
            torch.zeros(size_memory_per_process).long().to(self.device, non_blocking=True),
        )
        self.register_buffer(
            "local_memory_embeddings",
            F.normalize(
                torch.randn(self.num_large_crops, size_memory_per_process, self.proj_output_dim),
                dim=-1,
            ).to(self.device, non_blocking=True),
        )

    def on_train_epoch_start(self) -> None:
        """Prepares assigments and prototype centroids for the next epoch."""

        if self.current_epoch == 0:
            self.assignments = -torch.ones(len(self.num_prototypes), self.dataset_size, device=self.device).long()
        else:
            self.assignments, centroids = self.kmeans.cluster_memory(
                self.local_memory_index, self.local_memory_embeddings
            )
            for proto, centro in zip(self.prototypes, centroids):
                proto.weight.copy_(centro)

    def update_memory_banks(self, idxs: torch.Tensor, z: torch.Tensor, batch_idx: int) -> None:
        """Updates DeepClusterV2's memory banks of indices and features.

        Args:
            idxs (torch.Tensor): set of indices of the samples of the current batch.
            z (torch.Tensor): projected features of the samples of the current batch.
            batch_idx (int): batch index relative to the current epoch.
        """

        start_idx, end_idx = batch_idx * self.hparams.batch_size, (batch_idx + 1) * self.hparams.batch_size
        self.local_memory_index[start_idx:end_idx] = idxs
        for c, z_c in enumerate(z):
            self.local_memory_embeddings[c][start_idx:end_idx] = z_c.detach()

    def forward(self, X: torch.Tensor) -> Dict[str, Any]:
        """Performs the forward pass of the backbone, the projector and the prototypes.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]:
                a dict containing the outputs of the parent,
                the projected features and the logits.
        """

        feats = self.backbone(X)[0]
        logits = self.classifier(feats.detach())

        z = F.normalize(self.projector(feats))
        p = torch.stack([p(z) for p in self.prototypes])

        out = {
            "logits": logits,
            "feats": feats,
            "z": z,
            "p": p,
        }

        return out

    def _shared_step(self, X: torch.Tensor, targets: torch.Tensor) -> Dict:
        """Forwards a batch of images X and computes the classification loss, the logits, the
        features, acc@1 and acc@5.

        Args:
            X (torch.Tensor): batch of images in tensor format.
            targets (torch.Tensor): batch of labels for X.

        Returns:
            Dict: dict containing the classification loss, logits, features, acc@1 and acc@5.
        """

        out = self(X)
        logits = out["logits"]

        loss = F.cross_entropy(logits, targets, ignore_index=-1)
        # handle when the number of classes is smaller than 5
        top_k_max = min(5, logits.size(1))
        acc1, acc5 = accuracy_at_k(logits, targets, top_k=(1, top_k_max))

        out.update({"loss": loss, "acc1": acc1, "acc5": acc5})
        return out

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for DeepClusterV2 reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of DeepClusterV2 loss and classification loss.
        """
        idxs, X, targets = batch

        X = [X] if isinstance(X, torch.Tensor) else X

        outs = [self._shared_step(x, targets) for x in X[: self.num_large_crops]]
        outs = {k: [out[k] for out in outs] for k in outs[0].keys()}

        outs["loss"] = sum(outs["loss"]) / self.num_large_crops
        outs["acc1"] = sum(outs["acc1"]) / self.num_large_crops
        outs["acc5"] = sum(outs["acc5"]) / self.num_large_crops

        metrics = {
            "train_class_loss": outs["loss"],
            "train_acc1": outs["acc1"],
            "train_acc5": outs["acc5"],
        }

        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        class_loss = outs["loss"]
        z1, z2 = outs["z"]
        p1, p2 = outs["p"]

        # ------- deepclusterv2 loss -------
        preds = torch.stack([p1.unsqueeze(1), p2.unsqueeze(1)], dim=1)
        assignments = self.assignments[:, idxs]
        deepcluster_loss = deepclusterv2_loss_func(preds, assignments, self.temperature)

        # ------- update memory banks -------
        self.update_memory_banks(idxs, [z1, z2], batch_idx)

        self.log("train_deepcluster_loss", deepcluster_loss, on_epoch=True, sync_dist=True)

        return deepcluster_loss + class_loss

    def validation_step(self, batch: List[torch.Tensor], batch_idx: int, dataloader_idx: int = None) -> Dict[str, Any]:
        """Validation step for pytorch lightning. It does all the shared operations, such as
        forwarding a batch of images, computing logits and computing metrics.

        Args:
            batch (List[torch.Tensor]):a batch of data in the format of [img_indexes, X, Y].
            batch_idx (int): index of the batch.

        Returns:
            Dict[str, Any]: dict with the batch_size (used for averaging), the classification loss
                and accuracies.
        """

        X, targets = batch
        batch_size = targets.size(0)

        out = self._shared_step(X[0], targets)

        metrics = {
            "batch_size": batch_size,
            "val_loss": out["loss"],
            "val_acc1": out["acc1"],
            "val_acc5": out["acc5"],
        }
        return metrics

    def validation_epoch_end(self, outs: List[Dict[str, Any]]):
        """Averages the losses and accuracies of all the validation batches.
        This is needed because the last batch can be smaller than the others,
        slightly skewing the metrics.

        Args:
            outs (List[Dict[str, Any]]): list of outputs of the validation step.
        """

        val_loss = weighted_mean(outs, "val_loss", "batch_size")
        val_acc1 = weighted_mean(outs, "val_acc1", "batch_size")
        val_acc5 = weighted_mean(outs, "val_acc5", "batch_size")

        log = {"val_loss": val_loss, "val_acc1": val_acc1, "val_acc5": val_acc5}

        self.log_dict(log, sync_dist=True)
