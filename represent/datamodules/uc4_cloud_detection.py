import os
import sys
import torch
import warnings
import rasterio

import numpy as np
import pytorch_lightning as pl
import pandas as pd

from enum import Enum
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from argparse import ArgumentParser
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, Subset
from typing import Any, Callable, Dict, Optional, List
from torchtyping import TensorType
from torch.utils.data import DataLoader
from scipy.ndimage import zoom

from glob import glob
from represent.transforms.normalize import Sentinel2Normalize, IGBP2DFCNormalize
from represent.transforms.moco_transforms import ToTensorDirect
from represent.tools.utils import dataset_with_index

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
DataLoader.__module__ = "torch.utils.data"

# The dataset creation destroys the traditional band ordering meaning pretrained networks are no longer compatible
# Use the remapping to put the bands back in the correct order
WHUS2_CD_REMAP = (10, 0, 1, 2, 4, 5, 6, 3, 7, 11, 12, 8, 9)


class UC4CloudDetectionDataset(Dataset):
    def __init__(
        self,
        root: str,
        phase: str = "train",
        bin_size_for_stratification: int = 10,
        subset_fraction: float = 1.0,
        patch_size: int = 256,
        clustering: bool = False,
        *args,
        **kwargs,
    ):
        self.root = root
        self.phase = phase
        self.patch_size = patch_size
        self.clustering = clustering

        self.df = pd.read_csv(os.path.join(self.root, f"{phase}_set.csv"))
        self.df["klass"] = self.df.cloudiness.astype(int) // bin_size_for_stratification

        # Reduce the dataset to a smaller balanced version to see how the model performs
        if subset_fraction < 1.0:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - subset_fraction, random_state=42)
            train_idx, _ = next(sss.split(self.df, self.df.klass))
            self.df = self.df.iloc[train_idx]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        item = self.df.iloc[index]
        path_lbl = os.path.join(self.root, item.lbl_path)
        path_10m = path_lbl.replace("labels", "10m")
        path_20m = path_lbl.replace("labels", "20m")
        path_60m = path_lbl.replace("labels", "60m")

        with rasterio.open(path_lbl) as base_src:
            lbl = base_src.read()

            imgs = []
            with rasterio.open(path_10m) as src:
                imgs.append(src.read())

            with rasterio.open(path_20m) as src:
                im = zoom(src.read(), (1, 2, 2), order=0, mode="nearest", grid_mode=True)
                imgs.append(im)

            with rasterio.open(path_60m) as src:
                im = zoom(src.read(), (1, 6, 6), order=0, mode="nearest", grid_mode=True)
                imgs.append(im)

        img = np.concatenate(imgs, axis=0)[
            WHUS2_CD_REMAP,
        ]

        img = img.astype(np.float32) / 10000
        lbl = lbl.astype(np.int32) // 255

        # Randomly sample a patch
        h, w = img.shape[1:]

        imgs = []
        lbls = []

        n_crops = 2 if self.clustering else 1

        for k in range(n_crops):
            # TODO: Try change this to select other images of the same class as the augmentation for clustering
            x = np.random.randint(0, w - self.patch_size)
            y = np.random.randint(0, h - self.patch_size)

            i_c = img[:, y : y + self.patch_size, x : x + self.patch_size]
            l_c = lbl[:, y : y + self.patch_size, x : x + self.patch_size]

            imgs.append(torch.from_numpy(i_c).float())
            lbls.append(torch.from_numpy(l_c).float())

        if self.clustering:
            return (
                imgs,
                # We need the class label for clustering
                # We need to do the array hack as PyTorch crashes if we pass a single value
                torch.from_numpy(np.array([item.klass])).long()[0],
            )
        else:
            return imgs[0], lbls[0]


class UC4CloudDetectionDataModule(pl.LightningDataModule):
    name = "UC4_CD"

    def __init__(
        self,
        data_dir: str,
        seed: int,
        batch_size: int = 64,
        num_workers: int = 0,
        training_set_fraction: float = 0.8,
        limit_dataset: float = 1.0,
        patch_size: int = 256,
        clustering: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_frac = training_set_fraction
        self.limit_dataset_fraction = limit_dataset
        self.patch_size = patch_size
        self.clustering = clustering

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--data_dir", type=str, default="/data/RepreSent/UC5")
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--training_set_fraction", type=float, default=0.8)
        parser.add_argument("--patch_size", type=int, default=256)
        parser.add_argument("--clustering", action="store_true")
        return parser

    def _build_dataset_index_file(self, stage: str = "train"):
        dataset_stats = []
        files_lbls = glob(os.path.join(self.data_dir, stage, "labels", "*.tif"))

        for f in files_lbls:
            raster = rasterio.open(f)
            data = raster.read()

            fid = os.path.splitext(os.path.basename(f))[0]
            counts = np.bincount(data.ravel(), minlength=256)
            counts = (counts / counts.sum()) * 100

            dataset_stats.append(
                {
                    "fid": fid,
                    "lbl_path": os.path.relpath(f, self.data_dir),
                    "cloudiness": counts[255],
                    "clear": counts[128],
                    "nodata": counts[0],
                }
            )

        return pd.DataFrame(dataset_stats)

    def prepare_data(self) -> None:
        for stage in ["test", "train"]:
            if not os.path.exists(os.path.join(self.data_dir, f"{stage}_set.csv")):
                df_stats = self._build_dataset_index_file(stage)
                df_stats.to_csv(os.path.join(self.data_dir, f"{stage}_set.csv"), index=None)

    def setup(self, stage="fit"):
        if stage == "fit" or stage is None:
            self.dataset = UC4CloudDetectionDataset(
                self.data_dir,
                "train",
                bin_size_for_stratification=10,
                subset_fraction=self.limit_dataset_fraction,
                patch_size=self.patch_size,
                clustering=self.clustering,
            )

            # Split the dataset into 10 classes so we can stratify it into train and val
            classes = self.dataset.df.cloudiness.astype(int) // 10

            train_indices, val_indices = next(
                StratifiedShuffleSplit(
                    train_size=self.train_frac, test_size=1 - self.train_frac, n_splits=2, random_state=self.seed
                ).split(np.arange(len(classes)), classes)
            )

            if self.clustering:
                self.train_dataset = dataset_with_index(Subset)(self.dataset, train_indices)
            else:
                self.train_dataset = Subset(self.dataset, train_indices)

            self.val_dataset = Subset(self.dataset, val_indices)

        if stage == "test" or stage is None:
            self.test_dataset = UC4CloudDetectionDataset(
                self.data_dir,
                "test",
                patch_size=self.patch_size,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
