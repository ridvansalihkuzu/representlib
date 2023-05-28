import pytorch_lightning as pl
from pytorch_lightning.callbacks.finetuning import BaseFinetuning


class DelayedUnfreeze(BaseFinetuning):
    def __init__(
        self, backbone_id="backbone", unfreeze_at_epoch=10, train_frozen_bn: bool = True, reset_lr: float = None
    ):
        super().__init__()
        self.backbone_id = backbone_id
        self._unfreeze_at_epoch = unfreeze_at_epoch
        self._train_frozen_bn = train_frozen_bn
        self.unfrozen_lr = reset_lr

    def freeze_before_training(self, pl_module) -> None:
        backbone = getattr(pl_module, self.backbone_id)
        self.freeze(backbone, train_bn=self._train_frozen_bn)

    def finetune_function(self, pl_module, current_epoch, optimizer, optimizer_idx) -> None:
        if current_epoch == self._unfreeze_at_epoch:
            backbone = getattr(pl_module, self.backbone_id)

            if self.unfrozen_lr:
                optimizer.param_groups[0]["lr"] = float(self.unfrozen_lr)

            self.unfreeze_and_add_param_group(
                modules=backbone,
                optimizer=optimizer,
                train_bn=True,
                lr=self.unfrozen_lr,
            )
