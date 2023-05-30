import os
from cv2 import transform
import torch
import wandb

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from argparse import ArgumentParser
from datetime import datetime
from pytorch_lightning import seed_everything
from pl_bolts.metrics import mean, precision_at_k, accuracy
from pl_bolts.models.self_supervised import resnets
from represent.tools.utils import seed_all
from represent.tools.plots import (
    plot_confusion_matrix,
    print_classification_report,
)


class UC2ResNet(pl.LightningModule):
    def __init__(
        self,
        num_classes: int = 26,
        backbone: str = "resnet18",
        input_ch: int = 13,
        use_mlp: bool = False,
        classification_head: nn.Module = None,
        class_weights: torch.Tensor = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["classification_head"])

        self.num_classes = num_classes

        # Create the base model
        template_model = getattr(resnets, backbone)
        self.model = template_model(num_classes=num_classes)

        self.model.conv1 = nn.Conv2d(input_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Initialize this correctly
        nn.init.kaiming_normal_(self.model.conv1.weight, mode="fan_out", nonlinearity="relu")

        dim_mlp = self.model.fc.weight.shape[1]

        # Don't include the classification head - we can't remove it as the base model breaks so we just make it an identity transform
        if classification_head:
            self.model.fc = nn.Identity()

        if use_mlp:
            self.model.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp),
                nn.ReLU(),
                self.model.fc,
            )

        self.classifier = classification_head
        self.class_weights = class_weights

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
        y_pred = self.model(x)[0]
        y_pred = self.model.fc(y_pred)

        if self.classifier:
            y_pred = self.classifier(y_pred)

        return y_pred

    def on_train_start(self):
        if self.class_weights == "datamodule":
            train_ds = self.trainer.datamodule.train_dataloader().dataset
            if hasattr(train_ds, "get_class_weights") and callable(getattr(train_ds, "get_class_weights")):
                self.class_weights = torch.from_numpy(train_ds.get_class_weights()).float()
                print("Class Weights Loaded")

    def _common_step(
        self,
        batch,
        batch_idx,
        include_preds=False,
    ):
        x, y_true = batch
        y_pred = self(x)

        if self.training and self.class_weights is not None:
            # Only weight the loss for the training set not validation or test
            loss = F.cross_entropy(y_pred, y_true, weight=self.class_weights.to(y_pred.device))
        else:
            loss = F.cross_entropy(y_pred, y_true)

        log = {
            "loss": loss,
            "acc": accuracy(y_pred, y_true),
        }

        if include_preds:
            log["preds"] = y_pred.argmax(1).detach().cpu()
            log["target"] = y_true.detach().cpu()

        return log

    def training_step(self, batch, batch_idx):
        log = self._common_step(batch, batch_idx, False)
        return log

    def training_epoch_end(self, outputs):
        train_loss = mean(outputs, "loss")
        train_acc = mean(outputs, "acc")

        log = {"train_loss": train_loss, "train_acc": train_acc}
        self.log_dict(log)

    def validation_step(self, batch, batch_idx):
        log = self._common_step(batch, batch_idx, True)
        return log

    def validation_epoch_end(self, outputs):
        val_loss = mean(outputs, "loss")
        val_acc = mean(outputs, "acc")

        log = {"val_loss": val_loss, "val_acc": val_acc}
        self.log_dict(log)

        preds = torch.cat([tmp["preds"] for tmp in outputs])
        targets = torch.cat([tmp["target"] for tmp in outputs])

        if isinstance(self.trainer.logger, pl.loggers.WandbLogger):
            self.trainer.logger.experiment.log(
                {
                    "conf_mat": wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=targets.numpy(),
                        preds=preds.numpy(),
                        class_names=np.arange(self.num_classes),
                    )
                }
            )

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        log = self._common_step(batch, batch_idx, True)
        return log

    def test_step(self, batch, batch_idx):
        log = self._common_step(batch, batch_idx, True)
        return log

    def test_epoch_end(self, outputs):
        test_loss = mean(outputs, "loss")
        test_acc = mean(outputs, "acc")

        log = {"test_loss": test_loss, "test_acc": test_acc}
        self.log_dict(log)

        preds = torch.cat([tmp["preds"] for tmp in outputs]).numpy()
        targets = torch.cat([tmp["target"] for tmp in outputs]).numpy()

        print_classification_report(targets, preds)
        plot_confusion_matrix(targets, preds)

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
        parser.add_argument("--use_mlp", action="store_true")
        parser.add_argument("--classification_head", type=str, default=None)
        parser.add_argument("--learning_rate", type=float, default=1e-4)
        parser.add_argument("--weight_decay", type=float, default=1e-8)
        return parser


def cli_main():
    from represent.datamodules.uc2_landcover_datamodule import (
        UC2LandCoverDataModule,
        KEEP_CLASSES,
    )
    from represent.transforms.augmentations import get_data_augmentation
    from pytorch_lightning.loggers import WandbLogger

    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed to initialize with")
    parser.add_argument(
        "--model_ckpt",
        type=str,
        default=None,
        help="Model checkpoint path to initialize ResNet",
    )
    parser.add_argument(
        "--model_key",
        type=str,
        default=None,
        help="The key in the checkpoint file to load the ResNet from",
    )
    parser = pl.Trainer.add_argparse_args(parser)
    parser = UC2ResNet.add_model_specific_args(parser)
    parser = UC2LandCoverDataModule.add_model_specific_args(parser)
    args = parser.parse_args()

    seed_all(args.seed)

    ts = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    run_name = f"UC2_ResNet_{ts}"
    logger = WandbLogger(project="RepreSent", name=run_name, log_model=True, save_code=True)

    checkpointer = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(os.getcwd(), run_name),
        filename="{epoch}-{val_acc:.2f}",
        monitor="val_acc",
        mode="max",
        save_last=True,
    )

    early_stopping_callback = pl.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=10)

    args.keep_classes = KEEP_CLASSES
    uc2_datamodule = UC2LandCoverDataModule.from_argparse_args(args, transform=get_data_augmentation(cropsize=48))

    if args.classification_head == "linear":
        args.classification_head = nn.Linear(2048, args.num_classes)

    model = UC2ResNet(**args.__dict__)

    if args.model_ckpt:
        model.load_from_checkpoint(args.model_ckpt, filter_and_remap=args.model_key)

    trainer = pl.Trainer.from_argparse_args(args, enable_checkpointing=True, callbacks=[checkpointer], logger=logger)

    trainer.fit(model, datamodule=uc2_datamodule)


if __name__ == "__main__":
    cli_main()
