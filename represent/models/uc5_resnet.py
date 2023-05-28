import os
import torch
import wandb

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from typing import Union, List, Iterable
from argparse import ArgumentParser
from datetime import datetime
from pytorch_lightning import seed_everything
from pl_bolts.metrics import mean, precision_at_k
from torchmetrics.functional import r2_score, mean_squared_error
from pl_bolts.models.self_supervised import resnets
from represent.tools.utils import seed_all
from represent.tools.plots import (
    plot_confusion_matrix,
    print_classification_report,
)


class UC5ResNet(pl.LightningModule):
    def __init__(
        self,
        backbone: str = "resnet18",
        input_ch: Union[List[int], int] = 13,
        use_mlp: bool = False,
        classification_head: nn.Module = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["classification_head"])

        # Create the base model
        template_model = getattr(resnets, backbone)
        input_ch = input_ch if isinstance(input_ch, Iterable) else [input_ch]

        self.streams = nn.ModuleList()

        for ch in input_ch:
            model = template_model(num_classes=1)

            model.conv1 = nn.Conv2d(ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
            # Initialize this correctly
            nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")

            dim_mlp = model.fc.weight.shape[1]

            # Remove the built in classification head as we want to do regression
            model.fc = nn.Identity()

            if use_mlp:
                model.fc = nn.Sequential(
                    nn.Linear(dim_mlp, dim_mlp),
                    nn.ReLU(),
                    model.fc,
                )

            self.streams.append(model)

        self.classifier = classification_head

    def load_from_checkpoint(
        self,
        checkpoint: str,
        filter_and_remap: str = None,
        stream_id: int = 0,
        input_ch_subset: List[int] = None,
    ):
        ckpt = torch.load(checkpoint)

        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

        if filter_and_remap:
            state_dict = {k.replace(filter_and_remap, ""): v for k, v in state_dict.items() if filter_and_remap in k}

        if input_ch_subset:
            state_dict["conv1.weight"] = state_dict["conv1.weight"][:, input_ch_subset]

        return self.streams[stream_id].load_state_dict(state_dict, strict=False)

    def forward(self, xs):
        preds = []

        if isinstance(xs, list):
            assert len(xs) == len(self.streams), "Insufficient inputs or models"
        else:
            assert len(self.streams) == 1, "Insufficient inputs or models"

        for model, x in zip(self.streams, xs):
            y_pred = model(x.float())[0]
            y_pred = model.fc(y_pred)
            preds.append(y_pred)

        y_pred = torch.cat(preds, 1)

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

        loss = F.mse_loss(y_pred, y_true)

        log = {
            "loss": loss,
        }

        if include_preds:
            log["preds"] = y_pred.detach().cpu()
            log["target"] = y_true.detach().cpu()

        return log

    def training_step(self, batch, batch_idx):
        log = self._common_step(batch, batch_idx, True)
        return log

    def training_epoch_end(self, outputs):
        train_loss = mean(outputs, "loss")

        preds = torch.cat([tmp["preds"] for tmp in outputs])
        targets = torch.cat([tmp["target"] for tmp in outputs])

        train_r2 = r2_score(preds, targets)
        train_err = mean_squared_error(preds, targets, squared=False)

        log = {"train_loss": train_loss, "train_rmse": train_err, "train_r2": train_r2}
        self.log_dict(log)

    def validation_step(self, batch, batch_idx):
        log = self._common_step(batch, batch_idx, True)
        return log

    def validation_epoch_end(self, outputs):
        val_loss = mean(outputs, "loss")

        preds = torch.cat([tmp["preds"] for tmp in outputs])
        targets = torch.cat([tmp["target"] for tmp in outputs])

        val_r2 = r2_score(preds, targets)
        val_err = mean_squared_error(preds, targets, squared=False)

        log = {"val_loss": val_loss, "val_rmse": val_err, "val_r2": val_r2}
        self.log_dict(log)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        log = self._common_step(batch, batch_idx, True)
        return log

    def test_step(self, batch, batch_idx):
        log = self._common_step(batch, batch_idx, True)
        return log

    def test_epoch_end(self, outputs):
        test_loss = mean(outputs, "loss")

        log = {
            "test_loss": test_loss,
        }
        self.log_dict(log)

        preds = torch.cat([tmp["preds"] for tmp in outputs]).numpy()
        targets = torch.cat([tmp["target"] for tmp in outputs]).numpy()

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
        parser.add_argument("--input_ch", type=list, nargs="+", default=[2, 13])
        parser.add_argument("--use_mlp", action="store_true")
        parser.add_argument("--classification_head", type=str, default=None)
        parser.add_argument("--learning_rate", type=float, default=1e-4)
        parser.add_argument("--weight_decay", type=float, default=1e-8)
        return parser


def cli_main():
    pass
    # from represent.datamodules.uc2_landcover_datamodule import (
    #     UC2LandCoverDataModule,
    #     KEEP_CLASSES,
    # )
    # from represent.transforms.augmentations import get_data_augmentation
    # from pytorch_lightning.loggers import WandbLogger

    # parser = ArgumentParser()
    # parser.add_argument("--seed", type=int, default=42, help="Random seed to initialize with")
    # parser.add_argument(
    #     "--model_ckpt",
    #     type=str,
    #     default=None,
    #     help="Model checkpoint path to initialize ResNet",
    # )
    # parser.add_argument(
    #     "--model_key",
    #     type=str,
    #     default=None,
    #     help="The key in the checkpoint file to load the ResNet from",
    # )
    # parser = pl.Trainer.add_argparse_args(parser)
    # parser = UC2ResNet.add_model_specific_args(parser)
    # parser = UC2LandCoverDataModule.add_model_specific_args(parser)
    # args = parser.parse_args()

    # seed_all(args.seed)

    # ts = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    # run_name = f"UC2_ResNet_{ts}"
    # logger = WandbLogger(project="RepreSent", name=run_name, log_model=True, save_code=True)

    # checkpointer = pl.callbacks.ModelCheckpoint(
    #     dirpath=os.path.join(os.getcwd(), run_name),
    #     filename="{epoch}-{val_acc:.2f}",
    #     monitor="val_acc",
    #     mode="max",
    #     save_last=True,
    # )

    # early_stopping_callback = pl.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=10)

    # args.keep_classes = KEEP_CLASSES
    # uc2_datamodule = UC2LandCoverDataModule.from_argparse_args(args, transform=get_data_augmentation(cropsize=48))

    # if args.classification_head == "linear":
    #     args.classification_head = nn.Linear(2048, args.num_classes)

    # model = UC2ResNet(**args.__dict__)

    # if args.model_ckpt:
    #     model.load_from_checkpoint(args.model_ckpt, filter_and_remap=args.model_key)

    # trainer = pl.Trainer.from_argparse_args(args, enable_checkpointing=True, callbacks=[checkpointer], logger=logger)

    # trainer.fit(model, datamodule=uc2_datamodule)


if __name__ == "__main__":
    cli_main()
