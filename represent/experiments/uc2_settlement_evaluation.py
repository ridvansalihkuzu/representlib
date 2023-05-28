import sys

import pandas as pd

sys.path.append("../..")

from represent.datamodules.uc2_settlement_module import get_center, load_uc2_settlement_data

from meteor import models

import torch
import torchvision

from torch.nn import MaxPool2d, Identity
from collections import OrderedDict
import torch.nn.functional as F
from meteor import update_parameters
from sklearn.metrics import average_precision_score, jaccard_score
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from copy import copy

from argparse import ArgumentParser
import pytorch_lightning as pl

from represent.datamodules.uc2_landcover_datamodule import UC2LandCoverDataModule
from represent.models.uc2_segmentation_resnet import UC4ResNet
from skimage.morphology import binary_dilation, disk
import matplotlib.pyplot as plt
from skimage.exposure import equalize_hist

import os
import rasterio as rio

from rasterio.windows import Window
from rasterio.windows import from_bounds
from collections import OrderedDict

MOCO_CHECKPOINT = "../weights/uc2_moco_resnet18.pth"
SUPERVISED_CHECKPOINT = "../weights/uc2_supervised_resnet18.pth"
RANDOM_FOREST_RESULTS_PATH = "/data/RepreSent/UC2/uc2_settlements/random_forest/37LBL.tif"
RF_URBAN_CLASSID = 7


def main():

    num_shots = (200, 600)
    imagesize = 10240
    target_index = 600 # 600:test 200:validation
    threshold = None #0.5
    building_dilation_radius = 5

    datapath = "/data/RepreSent/UC2"
    savepath = f"/data/RepreSent/UC2/uc2_settlements/{target_index}"

    use_cache = False

    X, Y, x_test, existing_labels, buildings, meta = load_uc2_settlement_data(datapath=datapath,
                                                                     target_index=target_index,
                                                                     savepath=savepath,
                                                                     num_shots=num_shots,
                                                                     imagesize=imagesize,
                                                                     use_cache=True)

    # dilate buildings footprints to get a rough estimate of settlement locations (buildings are only single pixels)
    buildings = binary_dilation(buildings, footprint=disk(building_dilation_radius))

    logger = Logger()

    ## Supervised
    if not use_cache:
        probability_sup = run_supervised(X, Y, x_test)
        write_probability_tif(probability_sup, meta, os.path.join(savepath, "probability_sup.tif"))
    else:
        probability_sup = read_tif(os.path.join(savepath, "probability_sup.tif"))
    ap, iou, optimal_threshold = evaluate(probability_sup, buildings, mask=existing_labels, threshold=threshold)
    logger.log(name="supervised", ap=ap, iou=iou, threshold=optimal_threshold)

    ## MOCO
    if not use_cache:
        probability_moco = run_moco(X, Y, x_test)
        write_probability_tif(probability_moco, meta, os.path.join(savepath, "probability_moco.tif"))
    else:
        probability_moco = read_tif(os.path.join(savepath, "probability_moco.tif"))
    ap, iou, optimal_threshold = evaluate(probability_moco, buildings, mask=existing_labels, threshold=threshold)
    logger.log(name="moco", ap=ap, iou=iou, threshold=optimal_threshold)

    ## Random Init
    if not use_cache:
        probability_rand = run_randominit(X, Y, x_test)
        write_probability_tif(probability_rand, meta, os.path.join(savepath, "probability_rand.tif"))
    else:
        probability_rand = read_tif(os.path.join(savepath, "probability_rand.tif"))
    ap, iou, optimal_threshold = evaluate(probability_rand, buildings, mask=existing_labels, threshold=threshold)
    logger.log(name="random_init", ap=ap, iou=iou, threshold=optimal_threshold)

    ## MAML
    if not use_cache:
        probability_maml = run_maml(X, Y, x_test)
        write_probability_tif(probability_maml, meta, os.path.join(savepath, "probability_maml.tif"))
    else:
        probability_maml = read_tif(os.path.join(savepath, "probability_maml.tif"))
    ap, iou, optimal_threshold = evaluate(probability_maml, buildings, mask=existing_labels, threshold=threshold)
    logger.log(name="maml", ap=ap, iou=iou, threshold=optimal_threshold)

    ## Random Forest
    prediction_rf = read_tif(RANDOM_FOREST_RESULTS_PATH, meta=meta) == RF_URBAN_CLASSID
    ap, iou, optimal_threshold = evaluate(prediction_rf.astype("float"), buildings, mask=existing_labels, threshold=0.5)
    logger.log(name="random forest", ap=ap, iou=iou, threshold=0.5)

    fig = plot_results(x_test, existing_labels, buildings, prediction_rf, probability_maml, probability_moco, probability_rand, threshold=None)
    fig.savefig(os.path.join(savepath, "result.png"), bbox_inches="tight", transparent=True)
    print(f"writing qualitative results to {os.path.join(savepath, 'result.png')}")

    df = logger.dataframe()

    print(df.set_index("name"))#[["ap", "iou"]])

    plt.show()
class Logger():
    def __init__(self):
        self.stats = []

    def log(self, **kwargs):
        self.stats.append(dict(**kwargs))

    def dataframe(self):
        return pd.DataFrame(self.stats)

def read_tif(savepath, meta=None):

    with rio.open(savepath, "r") as src:

        if meta is not None:
            transform = meta["transform"]

            left = transform.xoff
            right = transform.xoff + transform.a * meta["width"]
            top = transform.yoff
            bottom = transform.yoff + transform.e * meta["height"]

            window = from_bounds(left, bottom, right, top, transform=src.transform)
        else:
            window = None

        arr = src.read(1, window=window)
    return arr
def write_probability_tif(probability, meta, savepath):
    profile = copy(meta)
    profile.update(count=1, dtype="float32")

    with rio.open(savepath, "w", **profile) as dst:
        dst.write(probability, 1)


def evaluate(probabiliy, buildings, mask=None, threshold=None):
    def find_optimal_threshold(y_true, y_score):
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        return optimal_threshold

    def metrics(y_true, y_score, optimal_threshold=None):
        ap = average_precision_score(y_true, y_score)

        if optimal_threshold is None:
            optimal_threshold = find_optimal_threshold(y_true, y_score)
        iou = jaccard_score(y_true=y_true, y_pred=(y_score > optimal_threshold), average='binary')

        return ap, iou, optimal_threshold

    buildings_flat = buildings.reshape(-1)
    probabiliy_flat = probabiliy.reshape(-1)
    mask_flat = mask.astype(bool).reshape(-1)

    return metrics(y_true=buildings_flat[~mask_flat], y_score=probabiliy_flat[~mask_flat], optimal_threshold=threshold)

def run_moco(X, Y, x_test):
    state_dict = torch.load(MOCO_CHECKPOINT)["state_dict"]
    state_dict_new = OrderedDict({k:v for k,v in state_dict.items() if "classifier" not in k})
    return run_finetune(X, Y, x_test, state_dict=state_dict_new)

def run_supervised(X, Y, x_test):
    state_dict = torch.load(SUPERVISED_CHECKPOINT)["state_dict"]
    state_dict_new = OrderedDict({k:v for k,v in state_dict.items() if "classifier" not in k})
    return run_finetune(X, Y, x_test, state_dict=state_dict_new)

def run_randominit(X, Y, x_test):
    return run_finetune(X, Y, x_test, state_dict=None)

def run_finetune(X, Y, x_test, state_dict):

    learning_rate = 1e-3
    epochs = 5
    batch_size = 32

    class SettlementFinetuneDataset(Dataset):
        def __init__(self, X, Y, transform=None):
            self.X = X
            self.Y = Y
            self.transform = transform

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            image = self.X[idx]
            label = self.Y[idx]
            if self.transform is not None:
                image = self.transform(image)

            return image, label

    ds = SettlementFinetuneDataset(X, Y)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    config = dict(
        # DataModule Settings
        data_dir=None,
        seed=42,
        batch_size=4,
        num_workers=16,
        training_set_fraction=0.70,
        limit_dataset=0.5,
        patch_size=256,

        # Trainer Settings
        gpus=1,
        accelerator="gpu",
        strategy=None,

        # Model Parameters
        use_mlp=False,
        num_classes=1,
        input_ch=13,
        backbone="resnet18",
        segmentation=True,
        checkpoint=None,

        # Optimizer Parameteres
        optimizer="Adam",
        scheduler="CosineAnnealingLR",
        momentum=0.9,
        max_epochs=50,
        learning_rate=2e-2,
    )

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = UC4ResNet.add_model_specific_args(parser)

    args, arg_strings = parser.parse_known_args([], None)
    for key, value in config.items():
        setattr(args, key, value)

    # args.classification_head = torch.nn.Linear(512, config['num_classes'])
    args.classification_head = torch.nn.Sequential(
        torch.nn.Conv2d(512, config['num_classes'], kernel_size=3, padding=1)
    )

    model = UC4ResNet(**args.__dict__)

    if state_dict is not None:
        model.load_state_dict(state_dict, strict=False)

    device = 'cuda'
    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    for _ in tqdm(range(epochs), total=epochs):
        model.to(device)

        losses = []
        for idx, batch in enumerate(dl):
            optimizer.zero_grad()

            x, y = batch

            y_pred = model(x.to(device))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(y_pred.squeeze(), y.to(device).float())
            loss.backward()
            optimizer.step()
            losses.append(loss.cpu().detach().numpy())

    X_trg = torch.from_numpy(x_test) * 1e-4
    with torch.no_grad():
        model.eval()
        p = torch.sigmoid(model(X_trg.unsqueeze(0).cuda()))

    return p.squeeze().cpu().numpy()

def run_maml(X, Y, x_test):

    subset_bands = ["S2B1", "S2B2", "S2B3", "S2B4", "S2B5", "S2B6", "S2B7", "S2B8", "S2B8A", "S2B9", "S2B10", "S2B11",
                    "S2B12"]

    # initialize an RGB model
    basemodel = models.get_model("maml_resnet12", subset_bands=subset_bands, segmentation=True)

    # basemodel.layer1[0].maxpool = Identity()
    # basemodel.layer2[0].maxpool = Identity()
    # basemodel.layer3[0].maxpool = MaxPool2d(kernel_size=[2, 2], stride=1, padding=0, dilation=1, ceil_mode=False)
    # basemodel.layer4[0].maxpool = MaxPool2d(kernel_size=[2, 2], stride=1, padding=0, dilation=1, ceil_mode=False)

    taskmodel = basemodel

    taskmodel.zero_grad()
    batch_size = 32
    device = "cuda"
    first_order = True
    inner_step_size = 0.2
    gradient_steps = 500
    verbose = False

    transforms = get_transform()

    taskmodel = taskmodel.to(device)

    param = OrderedDict(taskmodel.meta_named_parameters())
    for t in range(gradient_steps):

        idxs = np.random.randint(X.shape[0], size=batch_size)
        image = X[idxs].float()
        mask = Y[idxs].float()

        image, mask = transforms(image, mask)

        train_logit = taskmodel(image.to(device), params=param)

        inner_loss = F.binary_cross_entropy_with_logits(train_logit.squeeze(1), mask.to(device))
        param = update_parameters(taskmodel, inner_loss, params=param,
                                  inner_step_size=inner_step_size, first_order=first_order)

        if verbose:
            train_logit = taskmodel(image.to(device), params=param)
            loss_after_adaptation = F.binary_cross_entropy_with_logits(train_logit.squeeze(1),
                                                                       mask.to(device))
            print(
                f"adapting to class 1 with {X.shape[0]} samples: step {t}/{gradient_steps}: support loss {inner_loss:.2f} -> {loss_after_adaptation:.2f}")

    taskmodel = taskmodel.eval()

    X_trg = torch.from_numpy(x_test) * 1e-4

    with torch.no_grad():
        taskmodel.eval()
        p = torch.sigmoid(taskmodel(X_trg.unsqueeze(0).cuda(), params=param))

    return p.squeeze().cpu().numpy()
def get_transform():
    blur = torchvision.transforms.GaussianBlur(7)

    def transforms(image, mask):
        # augmentations
        if torch.rand(1) > 0.5:
            image = torch.fliplr(image.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            mask = torch.fliplr(mask)

        if torch.rand(1) > 0.5:
            image = torch.flipud(image.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            mask = torch.flipud(mask)

        if torch.rand(1) > 0.5:
            image = blur(image.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        return image, mask

    return transforms


def plot_results(x_test, existing_labels, buildings, prediction_rf, probability_maml, probability_moco, probability_randominit, threshold=None):

    fig, axs = plt.subplots(1,7, figsize=(7*3,3))

    ax = axs[0]
    ax.imshow(equalize_hist(x_test[np.array([3, 2, 1])]).transpose(1, 2, 0))
    ax.set_title("sentinel 2")

    ax = axs[1]
    ax.imshow(existing_labels)
    ax.set_title("existing labels")

    ax = axs[2]
    buildings_masked = buildings.astype("float")
    buildings_masked[existing_labels.astype(bool)] = np.nan
    ax.imshow(buildings_masked)
    ax.set_title("missed settlements")

    ax = axs[3]
    prediction_rf_masked = prediction_rf.astype("float")
    prediction_rf_masked[existing_labels.astype(bool)] = np.nan
    ax.imshow(prediction_rf_masked)
    ax.set_title("RF prediction")

    ax = axs[4]
    probability_maml = (probability_maml > 0.5) if threshold is not None else probability_maml
    probability_maml_masked = probability_maml.astype("float")
    probability_maml_masked[existing_labels.astype(bool)] = np.nan
    ax.imshow(probability_maml_masked)
    ax.set_title("MAML prediction")

    ax = axs[5]
    probability_moco = (probability_moco > 0.5) if threshold is not None else probability_moco
    probability_moco_masked = probability_moco.astype("float")
    probability_moco_masked[existing_labels.astype(bool)] = np.nan
    ax.imshow(probability_moco_masked)
    ax.set_title("MOCO prediction")

    ax = axs[6]
    probability_randominit = (probability_randominit > 0.5) if threshold is not None else probability_moco
    probability_randominit_masked = probability_randominit.astype("float")
    probability_randominit_masked[existing_labels.astype(bool)] = np.nan
    ax.imshow(probability_randominit_masked)
    ax.set_title("random init prediction")

    for ax in axs:
        ax.axis("off")

    plt.tight_layout()

    return fig

if __name__ == '__main__':
    main()
