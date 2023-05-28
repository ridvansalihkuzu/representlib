import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
from torchvision.models import resnet18
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch

from datamodules.uc2_landcover_datamodule import UC2LandCoverTaskDataModule
from pytorch_lightning.loggers import WandbLogger

from pytorch_lightning import seed_everything
seed_everything(42, workers=True)

from collections import OrderedDict
from meteor.models import get_model
from meteor import update_parameters
import torch

from transforms.augmentations import get_data_augmentation



def main():
    uc2landcover = UC2LandCoverTaskDataModule(data_dir="/data/RepreSent/UC2", image_size=610, num_workers=8,
                                              batch_size=12, episodes_per_epoch=1000,
                                              transform=get_data_augmentation(cropsize=48))

    model = LitModel(num_classes=26)  # .load_from_checkpoint()
    wandb_logger = WandbLogger(project="represent-maml", log_model=True)
    wandb_logger.watch(model)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_accuracy", mode="max")

    trainer = pl.Trainer(accelerator='gpu', devices=1, check_val_every_n_epoch=4,
                         callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=15), checkpoint_callback],
                         fast_dev_run=False, logger=wandb_logger)
    trainer.fit(model, uc2landcover)


class LitModel(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

        subset_bands = ["S2B1", "S2B2", "S2B3", "S2B4", "S2B5", "S2B6", "S2B7", "S2B8", "S2B8A", "S2B9", "S2B10",
                        "S2B11", "S2B12"]

        self.model = get_model("maml_resnet12", subset_bands=subset_bands, pretrained=True)
        self.gradient_steps = 1
        self.ways = 4
        self.shots = 2
        self.inner_step_size = 0.4
        self.first_order = True
        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.ways))

    def forward(self, x, params=None):
        return self.model(x, params=params)

    def training_step(self, batch, batch_idx):
        train_inputs, train_targets, test_inputs, test_targets = batch

        outer_loss = torch.tensor(0., device=train_targets.device)
        accuracy = torch.tensor(0., device=train_targets.device)

        batch_size = train_targets.shape[0]
        for train_input, train_target, test_input, test_target in zip(train_inputs, train_targets, test_inputs, test_targets):

            selected_class = np.random.choice(torch.unique(train_target).cpu().numpy())
            train_target = (train_target == selected_class).float()
            test_target = (test_target == selected_class).float()

            meta_named_parameters = self.model.meta_named_parameters()
            params = OrderedDict(meta_named_parameters)

            for t in range(self.gradient_steps):
                train_logit = self(train_input, params=params)
                inner_loss = self.criterion(train_logit.squeeze(1), train_target)
                params = update_parameters(model=self.model, loss=inner_loss, params=params,
                                           inner_step_size=self.inner_step_size, first_order=self.first_order)

            test_logit = self(test_input, params=params)
            outer_loss += self.criterion(test_logit.squeeze(1), test_target)
            predictions = (torch.sigmoid(test_logit.squeeze(1)) > 0.5).cpu().detach()
            accuracy += (predictions.view(-1).cpu() == test_target.view(-1).cpu()).float().mean()

        outer_loss.div_(batch_size)
        accuracy.div_(batch_size)

        return outer_loss

    def training_epoch_end(self, loss):
        self.log("train_loss", torch.stack([l["loss"] for l in loss]).mean())


    @torch.enable_grad()
    def validation_step(self, batch, batch_idx):
        train_inputs, train_targets, test_inputs, test_targets = batch

        outer_loss = torch.tensor(0., device=train_targets.device)
        accuracy = torch.tensor(0., device=train_targets.device)

        batch_size = train_targets.shape[0]
        for train_input, train_target, test_input, test_target in zip(train_inputs, train_targets, test_inputs, test_targets):

            selected_class = np.random.choice(torch.unique(train_target).cpu().numpy())
            train_target = (train_target == selected_class).float()
            test_target = (test_target == selected_class).float()

            meta_named_parameters = self.model.meta_named_parameters()
            params = OrderedDict(meta_named_parameters)

            for t in range(self.gradient_steps):
                train_logit = self(train_input, params=params)
                inner_loss = self.criterion(train_logit.squeeze(1), train_target)
                params = update_parameters(model=self.model, loss=inner_loss, params=params,
                                           inner_step_size=self.inner_step_size, first_order=self.first_order)

            test_logit = self(test_input, params=params)
            outer_loss += self.criterion(test_logit.squeeze(1), test_target)
            predictions = (torch.sigmoid(test_logit.squeeze(1)) > 0.5).cpu().detach()
            accuracy += (predictions.view(-1).cpu() == test_target.view(-1).cpu()).float().mean()

        outer_loss.div_(batch_size)
        accuracy.div_(batch_size)

        return {"val_loss":outer_loss.cpu().detach(),
                "val_accuracy": accuracy}

    def validation_epoch_end(self, outputs):
        self.log("val_loss", torch.stack([tmp['val_loss'] for tmp in outputs]).mean())
        self.log("val_accuracy", torch.stack([tmp['val_accuracy'] for tmp in outputs]).mean())

    def test_step(self, batch, batch_idx):
        self.eval()
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("test_loss", loss)
        self.log("test_accuracy", (y_hat.argmax(1) == torch.from_numpy(y)).mean())
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-8)

def update_parameters(model, loss, params=None, inner_step_size=0.5, first_order=False):
    """Update the parameters of the model, with one step of gradient descent.
    Parameters
    ----------
    model : `MetaModule` instance
        Model.
    loss : `torch.FloatTensor` instance
        Loss function on which the gradient are computed for the descent step.
    inner_step_size : float (default: `0.5`)
        Step-size of the gradient descent step.
    first_order : bool (default: `False`)
        If `True`, use the first-order approximation of MAML.
    Returns
    -------
    params : OrderedDict
        Dictionary containing the parameters after one step of adaptation.
    """

    if params is None:
        meta_params = model.meta_parameters()
    else:
        meta_params = params
        meta_params_list = list(meta_params.items())

    grads = torch.autograd.grad(loss, meta_params.values(),
                                create_graph=not first_order)

    params = OrderedDict()
    for (name, param), grad in zip(meta_params_list, grads):

        # overwrite static step size with dynamically learned step size
        if hasattr(model, "learning_rates"):
            inner_step_size = model.learning_rates[name.replace('.', '-')]

        # perform manual gradient step
        params[name] = param - inner_step_size * grad

    return params

if __name__ == '__main__':
    main()
