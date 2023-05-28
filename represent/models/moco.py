import os
import torch
import torchvision

import pytorch_lightning as pl

from torch import nn
from torch.nn import functional as F
from datetime import datetime
from argparse import ArgumentParser
from typing import Union, Any
from pl_bolts.metrics import mean, precision_at_k
from pl_bolts.models.self_supervised import Moco_v2
from represent.transforms.moco_transforms import NullTransform
from represent.losses.functional import infoNCE
from represent.tools.utils import modify_choices, seed_all, remove_argument


class DualMoco(pl.LightningModule):
    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.s1_moco = CustomDepthMoco_v2(input_ch=2, *args, **kwargs)
        self.s2_moco = CustomDepthMoco_v2(input_ch=13, *args, **kwargs)

    def forward(self, img_q1, img_k1, img_q2, img_k2):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            encoded q and k
        """

        q1, k1 = self.s1_moco(img_q1, img_k1)
        q2, k2 = self.s2_moco(img_q2, img_k2)

        return q1, k1, q2, k2

    def training_step(self, batch, batch_idx):
        (img_q1, img_k1, img_q2, img_k2), _ = batch

        # Update the momentum encoders
        self.s1_moco._momentum_update_key_encoder()
        self.s2_moco._momentum_update_key_encoder()

        qs1, ks1, qs2, ks2 = self(img_q1=img_q1, img_k1=img_k1, img_q2=img_q2, img_k2=img_k2)

        # Compute the standard single modality MoCo losses
        loss_s1, target_s1, output_s1 = infoNCE(qs1, ks1, self.s1_moco.queue, self.hparams.softmax_temperature, True)
        loss_s2, target_s2, output_s2 = infoNCE(qs2, ks2, self.s2_moco.queue, self.hparams.softmax_temperature, True)

        # Compute the cross modality MoCo loss
        loss_s1s2, target_s1s2, output_s1s2 = infoNCE(
            qs1, ks2, self.s2_moco.queue, self.hparams.softmax_temperature, True
        )
        loss_s2s1, target_s2s1, output_s2s1 = infoNCE(
            qs2, ks1, self.s1_moco.queue, self.hparams.softmax_temperature, True
        )

        # Balanced loss
        loss = 0.25 * (loss_s1 + loss_s2 + loss_s1s2 + loss_s2s1)

        self.s1_moco._dequeue_and_enqueue(ks1, queue=self.s1_moco.queue, queue_ptr=self.s1_moco.queue_ptr)
        self.s2_moco._dequeue_and_enqueue(ks2, queue=self.s2_moco.queue, queue_ptr=self.s2_moco.queue_ptr)

        acc1_s1, acc5_s1 = precision_at_k(output_s1, target_s1, top_k=(1, 5))
        acc1_s2, acc5_s2 = precision_at_k(output_s2, target_s2, top_k=(1, 5))
        acc1_s1s2, acc5_s1s2 = precision_at_k(output_s1s2, target_s1s2, top_k=(1, 5))
        acc1_s2s1, acc5_s2s1 = precision_at_k(output_s2s1, target_s2s1, top_k=(1, 5))

        log = {
            "train_loss": loss,
            "train_loss_s1": loss_s1,
            "train_loss_s2": loss_s2,
            "train_loss_s1s2": loss_s1s2,
            "train_loss_s2s1": loss_s2s1,
            "train_acc_s1": acc1_s1,
            "train_acc_s2": acc1_s2,
            "train_acc_s1s2": acc1_s1s2,
            "train_acc_s2s1": acc1_s2s1,
        }

        self.log_dict(log)
        return loss

    def validation_step(self, batch, batch_idx):
        (img_q1, img_k1, img_q2, img_k2), _ = batch

        qs1, ks1, qs2, ks2 = self(img_q1=img_q1, img_k1=img_k1, img_q2=img_q2, img_k2=img_k2)

        # Compute the standard single modality MoCo losses
        loss_s1, target_s1, output_s1 = infoNCE(
            qs1, ks1, self.s1_moco.val_queue, self.hparams.softmax_temperature, True
        )
        loss_s2, target_s2, output_s2 = infoNCE(
            qs2, ks2, self.s2_moco.val_queue, self.hparams.softmax_temperature, True
        )

        # Compute the cross modality MoCo loss
        loss_s1s2, target_s1s2, output_s1s2 = infoNCE(
            qs1, ks2, self.s2_moco.val_queue, self.hparams.softmax_temperature, True
        )
        loss_s2s1, target_s2s1, output_s2s1 = infoNCE(
            qs2, ks1, self.s1_moco.val_queue, self.hparams.softmax_temperature, True
        )

        # Balanced loss
        loss = 0.25 * (loss_s1 + loss_s2 + loss_s1s2 + loss_s2s1)

        self.s1_moco._dequeue_and_enqueue(ks1, queue=self.s1_moco.val_queue, queue_ptr=self.s1_moco.val_queue_ptr)
        self.s2_moco._dequeue_and_enqueue(ks2, queue=self.s2_moco.val_queue, queue_ptr=self.s2_moco.val_queue_ptr)

        acc1_s1, acc5_s1 = precision_at_k(output_s1, target_s1, top_k=(1, 5))
        acc1_s2, acc5_s2 = precision_at_k(output_s2, target_s2, top_k=(1, 5))
        acc1_s1s2, acc5_s1s2 = precision_at_k(output_s1s2, target_s1s2, top_k=(1, 5))
        acc1_s2s1, acc5_s2s1 = precision_at_k(output_s2s1, target_s2s1, top_k=(1, 5))

        log = {
            "val_loss": loss,
            "val_loss_s1": loss_s1,
            "val_loss_s2": loss_s2,
            "val_loss_s1s2": loss_s1s2,
            "val_loss_s2s1": loss_s2s1,
            "val_acc_s1": acc1_s1,
            "val_acc_s2": acc1_s2,
            "val_acc_s1s2": acc1_s1s2,
            "val_acc_s2s1": acc1_s2s1,
        }

        return log

    def training_epoch_end(self, outputs) -> None:
        return super().training_epoch_end(outputs)

    def validation_epoch_end(self, outputs):
        log = {}

        log["val_loss"] = mean(outputs, "val_loss")
        log["val_loss_s1"] = mean(outputs, "val_loss_s1")
        log["val_loss_s2"] = mean(outputs, "val_loss_s2")
        log["val_loss_s1s2"] = mean(outputs, "val_loss_s1s2")
        log["val_loss_s2s1"] = mean(outputs, "val_loss_s2s1")

        log["val_acc_s1"] = mean(outputs, "val_acc_s1")
        log["val_acc_s2"] = mean(outputs, "val_acc_s2")
        log["val_acc_s1s2"] = mean(outputs, "val_acc_s1s2")
        log["val_acc_s2s1"] = mean(outputs, "val_acc_s2s1")

        self.log_dict(log)

    def configure_optimizers(self):
        # This is dirty and not recommended, but it is safe only in this case where MoCo just needs to pull
        # attributes from the Trainer and never sets state.
        if not self.s1_moco.trainer:
            self.s1_moco.trainer = self.trainer

        if not self.s2_moco.trainer:
            self.s2_moco.trainer = self.trainer

        optimizer = torch.optim.SGD(
            self.parameters(),
            self.hparams.learning_rate,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            self.trainer.max_epochs,
        )

        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = CustomDepthMoco_v2.add_model_specific_args(parent_parser)
        remove_argument(parser, "--input_ch")

        return parser


class CustomDepthMoco_v2(Moco_v2):
    def init_encoders(self, base_encoder):
        template_model = getattr(torchvision.models, base_encoder)
        encoder_q = template_model(num_classes=self.hparams.emb_dim)
        encoder_k = template_model(num_classes=self.hparams.emb_dim)

        encoder_q.conv1 = nn.Conv2d(self.hparams.input_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        encoder_k.conv1 = nn.Conv2d(self.hparams.input_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)

        nn.init.kaiming_normal_(encoder_q.conv1.weight, mode="fan_out", nonlinearity="relu")
        nn.init.kaiming_normal_(encoder_k.conv1.weight, mode="fan_out", nonlinearity="relu")

        return encoder_q, encoder_k

    def forward(self, img_q, img_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(img_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # shuffle for making use of BN
            if self._use_ddp_or_ddp2(self.trainer):
                img_k, idx_unshuffle = self._batch_shuffle_ddp(img_k)

            k = self.encoder_k(img_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            if self._use_ddp_or_ddp2(self.trainer):
                k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        return q, k

    def training_step(self, batch, batch_idx):
        (img_1, img_2), _ = batch

        self._momentum_update_key_encoder()  # update the key encoder
        qs, ks = self(img_q=img_1, img_k=img_2)
        # Compute the InfoNCE loss
        loss, target, output = infoNCE(qs, ks, self.queue, self.hparams.softmax_temperature, True)

        self._dequeue_and_enqueue(ks, queue=self.queue, queue_ptr=self.queue_ptr)  # dequeue and enqueue

        acc1, acc5 = precision_at_k(output, target, top_k=(1, 5))

        log = {"train_loss": loss, "train_acc1": acc1, "train_acc5": acc5}
        self.log_dict(log)
        return loss

    def validation_step(self, batch, batch_idx):
        (img_1, img_2), _ = batch

        qs, ks = self(img_q=img_1, img_k=img_2)

        # Compute the InfoNCE loss
        loss, target, output = infoNCE(qs, ks, self.val_queue, self.hparams.softmax_temperature, True)

        # Update the queue
        self._dequeue_and_enqueue(ks, queue=self.val_queue, queue_ptr=self.val_queue_ptr)

        acc1, acc5 = precision_at_k(output, target, top_k=(1, 5))

        results = {"val_loss": loss, "val_acc1": acc1, "val_acc5": acc5}
        return results

    def validation_epoch_end(self, outputs):
        val_loss = mean(outputs, "val_loss")
        val_acc1 = mean(outputs, "val_acc1")
        val_acc5 = mean(outputs, "val_acc5")

        log = {"val_loss": val_loss, "val_acc1": val_acc1, "val_acc5": val_acc5}
        self.log_dict(log)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = Moco_v2.add_model_specific_args(parent_parser)
        parser.add_argument("--input_ch", type=int, default=3)
        parser.add_argument("--patch_sz", type=int, default=256)
        parser.add_argument("--band_set", type=str, default="all", choices=["all", "s1", "s2-all", "s2-reduced"])
        # We need to overwrite the existing dataset arguments so we first remove it
        modify_choices(parser, "dataset", ["SEN12MS"])

        return parser


def cli_main():
    from represent.datamodules.represent_sen12ms_datamodule import RepresentSEN12MSDataModule
    from represent.transforms.moco_transforms import MocoSEN12MSPacker
    from pytorch_lightning.loggers import WandbLogger

    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed to initialize with")
    parser = pl.Trainer.add_argparse_args(parser)
    parser = CustomDepthMoco_v2.add_model_specific_args(parser)
    args = parser.parse_args()

    seed_all(args.seed)

    ts = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    run_name = f"MoCo_Pretrain_{ts}"
    logger = WandbLogger(project="RepreSent", name=run_name, log_model=True, save_code=True)

    checkpointer = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(os.getcwd(), run_name),
        filename="{epoch}-{val_loss:.2f}",
        monitor="val_loss",
        save_last=True,
    )

    if args.dataset == "SEN12MS":
        # We leave the default transformations as they normalize the data into the [0, 1] range – we then used a data packer
        # to apply additional transformations and create the q,k pairs needed by MoCo. This is a little hacky but it works.
        # The limitations here are due to the SEN12MS dataset implementation in TorchGeo
        datamodule = RepresentSEN12MSDataModule.from_argparse_args(
            args,
            output_packer=MocoSEN12MSPacker(bands=args.band_set, patch_sz=args.patch_sz),
            s1_transform=NullTransform(),
            s2_transform=NullTransform(),
            lc_transform=NullTransform(),
        )
    else:
        raise ValueError("Only the SEN12MS dataset is supported currently.")

    if args.band_set == "all":
        params = args.__dict__
        del params["input_ch"]
        model = DualMoco(**params)
    else:
        model = CustomDepthMoco_v2(**args.__dict__)

    trainer = pl.Trainer.from_argparse_args(args, enable_checkpointing=True, callbacks=[checkpointer], logger=logger)
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    cli_main()
