import os
import torch
import wandb
import torchmetrics

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from argparse import ArgumentParser
from datetime import datetime
from pytorch_lightning import seed_everything
from pl_bolts.metrics import mean, precision_at_k, accuracy
from pl_bolts.models.self_supervised import resnets
from torchmetrics.classification import BinaryJaccardIndex, BinaryAccuracy

# from represent.tools.utils import seed_all, set_stride_recursive
from represent.tools.plots import (
    plot_confusion_matrix,
    print_classification_report,
)

from represent.losses.functional import soft_dice_loss_with_logits


def set_stride_recursive(module, stride):
    if hasattr(module, "stride"):
        module.stride = (stride, stride)

    for child in module.children():
        set_stride_recursive(child, stride)


class UC4ResNet(pl.LightningModule):
    def __init__(
        self,
        num_classes: int = 2,
        backbone: str = "resnet18",
        input_ch: int = 13,
        segmentation: bool = True,
        classification_head: nn.Module = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["classification_head"])

        self.num_classes = num_classes
        self.segmentation = segmentation

        # Create the base model
        template_model = getattr(resnets, backbone)
        self.model = template_model(num_classes=num_classes, return_all_feature_maps=self.segmentation)

        self.model.conv1 = nn.Conv2d(input_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Initialize this correctly
        nn.init.kaiming_normal_(self.model.conv1.weight, mode="fan_out", nonlinearity="relu")

        dim_mlp = self.model.fc.weight.shape[1]
        self.model.fc = nn.Identity()

        if self.segmentation:
            set_stride_recursive(self, 1)

        self.classifier = classification_head
        self.iou_metric_train = BinaryJaccardIndex()
        self.iou_metric_val = BinaryJaccardIndex()
        self.oa_train = BinaryAccuracy()
        self.oa_val = BinaryAccuracy()

    def load_from_checkpoint(
        self,
        checkpoint: str,
        filter_and_remap: str = None,
    ):
        ckpt = torch.load(checkpoint)

        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

        if filter_and_remap:
            state_dict = {
                k.replace(filter_and_remap, "model"): v for k, v in state_dict.items() if filter_and_remap in k
            }

        return self.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        # Get the last layer returned by the basemodel
        y_pred = self.model(x)[-1]

        if self.classifier:
            y_pred = self.classifier(y_pred)

        return y_pred

    def _common_step(
        self,
        batch,
        batch_idx,
        include_preds=False,
    ):
        x, y_true = batch
        y_pred = self(x)

        loss = soft_dice_loss_with_logits(y_pred, y_true) + F.binary_cross_entropy_with_logits(y_pred, y_true)

        return loss, y_pred, y_true

    def training_step(self, batch, batch_idx):
        loss, y_pred, y_true = self._common_step(batch, batch_idx, False)
        iou = self.iou_metric_train(y_pred, y_true)
        oa = self.oa_train(y_pred, y_true)

        log = {
            "loss": loss,
            "acc": oa,
            "iou": iou,
        }

        return log

    def training_epoch_end(self, outputs):
        train_loss = mean(outputs, "loss")

        log = {
            "train_loss": train_loss,
            "train_acc": self.oa_train.compute(),
            "train_iou": self.iou_metric_train.compute(),
        }

        self.log_dict(log)

    def validation_step(self, batch, batch_idx):
        loss, y_pred, y_true = self._common_step(batch, batch_idx, True)
        iou = self.iou_metric_val(y_pred, y_true)
        oa = self.oa_val(y_pred, y_true)

        log = {
            "loss": loss,
            "acc": oa,
            "iou": iou,
        }

        return log

    def validation_epoch_end(self, outputs):
        val_loss = mean(outputs, "loss")

        log = {
            "val_loss": val_loss,
            "val_acc": self.oa_val.compute(),
            "val_iou": self.iou_metric_val.compute(),
        }
        self.log_dict(log)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        loss, y_pred, y_true = self._common_step(batch, batch_idx, True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, y_pred, y_true = self._common_step(batch, batch_idx, True)
        return {"loss": loss}

    def test_epoch_end(self, outputs):
        test_loss = mean(outputs, "loss")

        log = {"test_loss": test_loss}
        self.log_dict(log)

        # preds = torch.cat([tmp["preds"] for tmp in outputs]).numpy()
        # targets = torch.cat([tmp["target"] for tmp in outputs]).numpy()

        # print_classification_report(targets, preds)
        # plot_confusion_matrix(targets, preds)

    def configure_optimizers(self):
        if "optimizer" in self.hparams and self.hparams.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.parameters()),
                self.hparams.learning_rate,
                momentum=self.hparams.momentum,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.parameters()),
                self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )

        if "scheduler" in self.hparams:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                self.trainer.max_epochs,
            )
        else:
            scheduler = None

        if scheduler:
            return [optimizer], [scheduler]
        else:
            return [optimizer]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--backbone", type=str, default="resnet18")
        parser.add_argument("--num_classes", type=int, default=21)
        parser.add_argument("--input_ch", type=int, default=13)
        parser.add_argument("--segmentation", action="store_true")
        parser.add_argument("--classification_head", type=str, default=None)
        parser.add_argument("--learning_rate", type=float, default=1e-4)
        parser.add_argument("--weight_decay", type=float, default=1e-8)
        return parser


# def cli_main():
#     from represent.datamodules.uc2_landcover_datamodule import (
#         UC2LandCoverDataModule,
#         KEEP_CLASSES,
#     )
#     from represent.transforms.augmentations import get_data_augmentation
#     from pytorch_lightning.loggers import WandbLogger

#     parser = ArgumentParser()
#     parser.add_argument("--seed", type=int, default=42, help="Random seed to initialize with")
#     parser.add_argument(
#         "--model_ckpt",
#         type=str,
#         default=None,
#         help="Model checkpoint path to initialize ResNet",
#     )
#     parser.add_argument(
#         "--model_key",
#         type=str,
#         default=None,
#         help="The key in the checkpoint file to load the ResNet from",
#     )
#     parser = pl.Trainer.add_argparse_args(parser)
#     parser = UC2ResNet.add_model_specific_args(parser)
#     parser = UC2LandCoverDataModule.add_model_specific_args(parser)
#     args = parser.parse_args()

#     seed_all(args.seed)

#     ts = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
#     run_name = f"UC2_ResNet_{ts}"
#     logger = WandbLogger(project="RepreSent", name=run_name, log_model=True, save_code=True)

#     checkpointer = pl.callbacks.ModelCheckpoint(
#         dirpath=os.path.join(os.getcwd(), run_name),
#         filename="{epoch}-{val_acc:.2f}",
#         monitor="val_acc",
#         mode="max",
#         save_last=True,
#     )

#     early_stopping_callback = pl.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=10)

#     args.keep_classes = KEEP_CLASSES
#     uc2_datamodule = UC2LandCoverDataModule.from_argparse_args(args, transform=get_data_augmentation(cropsize=48))

#     if args.classification_head == "linear":
#         args.classification_head = nn.Linear(2048, args.num_classes)

#     model = UC2ResNet(**args.__dict__)

#     if args.model_ckpt:
#         model.load_from_checkpoint(args.model_ckpt, filter_and_remap=args.model_key)

#     trainer = pl.Trainer.from_argparse_args(args, enable_checkpointing=True, callbacks=[checkpointer], logger=logger)

#     trainer.fit(model, datamodule=uc2_datamodule)


# if __name__ == "__main__":
#     cli_main()
