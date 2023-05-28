import os
import sys
import torch
import rasterio

import numpy as np
import albumentations as A
import pytorch_lightning as pl
import geopandas as gpd

from enum import Enum
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from argparse import ArgumentParser
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, Subset
from typing import Any, Callable, Dict, Optional, List
from torchtyping import TensorType
from torch.utils.data import DataLoader

from represent.transforms.normalize import Sentinel1Normalize, Sentinel2Normalize, IGBP2DFCNormalize
from represent.transforms.moco_transforms import ToTensorDirect

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
DataLoader.__module__ = "torch.utils.data"


class TimeSeriesReductions(Enum):
    NONE = "none"
    MEAN = "mean"
    STD = "std"
    MAX = "max"
    MIN = "min"


class UC5ForestBiomassDataset(Dataset):
    def __init__(
        self,
        root: str,
        patch_size_px: int = 24,
        reduce_ts: TimeSeriesReductions = TimeSeriesReductions.NONE,
        s1_transform: Optional[Callable] = None,
        s2_transform: Optional[Callable] = None,
        drop_ids: List[int] = [422954, 423324, 423332, 493381],  # These don't have enough pixels
        s2_level: str = "L1C",
        s2_only: bool = False,
    ):
        self.plots = gpd.read_file(os.path.join(root, "GT/plot_shapefiles/forest_plots.shp"))

        if drop_ids is not None:
            self.plots = self.plots[self.plots.PLOTID.isin(drop_ids) == False].reset_index()

        self.patch_px = patch_size_px
        self.reduce_ts = reduce_ts
        self.reducer = getattr(np, reduce_ts.value, lambda x, axis: x) or None
        self.s2_only = s2_only

        self.s1_transform = s1_transform
        self.s2_transform = s2_transform

        s2_fname = (
            "S2A_MSIL1C_20150817_10m.tif"
            if s2_level.upper() == "L1C"
            else "S2A_MSIL2A_20150817_10m_12_bands_no_SWIR.tif"
        )

        self.src_s2 = rasterio.open(os.path.join(root, "S2", s2_fname))

        if not s2_only:
            self.src_vh = rasterio.open(os.path.join(root, "S1", "recH50km_VH_2015_10m.tif"))
            self.src_vv = rasterio.open(os.path.join(root, "S1", "recH50km_VV_2015_10m.tif"))

        # Get the image resolution
        res = self.src_s2.res[0]

        self.plots["patch"] = self.plots["geometry"].centroid.buffer((self.patch_px) * res // 2)
        self.plots["class"] = (self.plots.H // 5).astype(int)

    def __len__(self):
        return len(self.plots)

    def __getitem__(self, index):
        plot = self.plots.iloc[index]
        bounds = plot["patch"].bounds

        if not self.s2_only:
            s1_vv = self.src_vv.read(window=from_bounds(*bounds, self.src_vv.transform))
            s1_vh = self.src_vh.read(window=from_bounds(*bounds, self.src_vh.transform))

            if self.reduce_ts == TimeSeriesReductions.NONE:
                raise Exception("Not yet implemented - we would want to return a list of tensors")
            else:
                s1_vv = self.reducer(s1_vv, 0)
                s1_vh = self.reducer(s1_vh, 0)

            s1_img = np.stack([s1_vv, s1_vh], axis=0)
            # s1_img = 10 * np.log10(s1_img / 65536)

            if self.s1_transform:
                s1_img = self.s1_transform(s1_img)

        s2_img = self.src_s2.read(window=from_bounds(*bounds, self.src_s2.transform))

        if self.s2_transform:
            s2_img = self.s2_transform(s2_img)
        else:
            s2_img = s2_img

        if self.s2_only:
            return (torch.from_numpy(s2_img).float(),), torch.Tensor([plot.H]).float()
        else:
            return (torch.from_numpy(s1_img).float(), torch.from_numpy(s2_img).float()), torch.Tensor([plot.H]).float()


class UC5ForestBiomassDataModule(pl.LightningDataModule):
    name = "UC5"

    def __init__(
        self,
        data_dir: str,
        seed: int,
        batch_size: int = 64,
        num_workers: int = 0,
        patch_size_px: int = 24,
        reduce_ts: TimeSeriesReductions = TimeSeriesReductions.MEAN,
        training_set_fraction: float = 0.8,
        s1_transform: Optional[Callable] = None,
        s2_transform: Optional[Callable] = None,
        s2_level: str = "L1C",
        s2_only: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.seed = seed
        self.batch_size = batch_size
        self.patch_size_px = patch_size_px
        self.num_workers = num_workers
        self.s1_transform = s1_transform
        self.s2_transform = s2_transform
        self.reduce_ts = TimeSeriesReductions(reduce_ts) if isinstance(reduce_ts, str) else reduce_ts
        self.train_frac = training_set_fraction
        self.s2_level = s2_level
        self.s2_only = s2_only

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--data_dir", type=str, default="/data/RepreSent/UC5")
        parser.add_argument("--s2_level", type=str, default="L1C")
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--patch_size_px", type=int, default=24)
        parser.add_argument("--s2_only", action="store_true")
        parser.add_argument(
            "--reduce_ts", type=TimeSeriesReductions, default="mean", choices=[r.value for r in TimeSeriesReductions]
        )
        parser.add_argument("--training_set_fraction", type=float, default=0.8)
        return parser

    def setup(self, stage="fit"):
        self.dataset = UC5ForestBiomassDataset(
            root=self.data_dir,
            patch_size_px=self.patch_size_px,
            reduce_ts=self.reduce_ts,
            s1_transform=self.s1_transform,
            s2_transform=self.s2_transform,
            s2_level=self.s2_level,
            s2_only=self.s2_only,
        )

        classes = self.dataset.plots["class"].values

        train_indices, test_indices = next(
            StratifiedShuffleSplit(
                train_size=self.train_frac, test_size=1 - self.train_frac, n_splits=2, random_state=self.seed
            ).split(np.arange(len(classes)), classes, groups=classes)
        )

        ti, vi = next(
            StratifiedShuffleSplit(train_size=0.8, test_size=0.2, n_splits=2, random_state=self.seed).split(
                train_indices, classes[train_indices], groups=classes[train_indices]
            )
        )

        train_indices, val_indices = train_indices[ti], train_indices[vi]

        if stage == "fit" or stage is None:
            self.train_dataset = Subset(self.dataset, train_indices)
            self.val_dataset = Subset(self.dataset, val_indices)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_dataset = Subset(self.dataset, test_indices)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
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
        )

    def predict_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )


if __name__ == "__main__":
    data_dir = "/data/RepreSent/UC5"
    dm = UC5ForestBiomassDataModule(
        data_dir, batch_size=32, patch_size_px=24, reduce_ts=TimeSeriesReductions.MEAN, training_set_fraction=0.6
    )
    dm.prepare_data()
    dm.setup()
    train_ds = dm.train_dataset

    import code

    code.interact(local=locals())
