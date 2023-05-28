import os
import sys
import torch

import numpy as np
import albumentations as A

from typing import Any, Callable, Dict, Optional
from torchtyping import TensorType
from torchgeo.datamodules import SEN12MSDataModule as BaseSEN12MSDataModule
from torch.utils.data import DataLoader

from represent.transforms.normalize import Sentinel1Normalize, Sentinel2Normalize, IGBP2DFCNormalize
from represent.transforms.moco_transforms import ToTensorDirect

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
DataLoader.__module__ = "torch.utils.data"


class RepresentSEN12MSDataModule(BaseSEN12MSDataModule):
    name = "SEN12MS"

    def __init__(
        self,
        data_dir: str,
        seed: int,
        band_set: str = "all",
        batch_size: int = 64,
        num_workers: int = 0,
        s1_transform: Optional[Callable] = None,
        s2_transform: Optional[Callable] = None,
        lc_transform: Optional[Callable] = None,
        output_packer: Optional[Callable] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        A modified SEN12MS dataloader for use with the RepreSent project

        Parameters
        ----------
        data_dir : str
            The root directory for the SEN12MS dataset - if the dataset does not exist at this directory it will be downloaded.
        seed : int
            Random seed for the dataloader.
        band_set : str, optional
            Can be either "all", "s1", "s2-all" or "s2-reduced", default is "all"
        batch_size : int, optional
            The batch size, default is 64
        num_workers : int, optional
            Number of workers loading the dataset, by default 0
        s1_transform : Optional[Callable], optional
            A function accepting and returning a tensor to transform the Sentinel-1 data, by default Sentinel1Normalize is used
        s2_transform : Optional[Callable], optional
            A function accepting and returning a tensor to transform the Sentinel-2 data, by default Sentinel2Normalize is used
        lc_transform : Optional[Callable], optional
            A function accepting and returning a tensor to transform the  IGBP landcover to DFC landcover, by default IGBP2DFCNormalize is used
        output_packer : Optional[Callable], optional
            A function accepting the transformed tensors which packs them into the format which your model expects, by default {'image': Tensor[S1,S2], 'mask': Tensor[DFC_LC]}
        """
        super().__init__(data_dir, seed, band_set, batch_size, num_workers, **kwargs)
        self.s1_transform = (
            A.Compose(
                [
                    Sentinel1Normalize(toDb=True, clip_to="nesz"),
                    ToTensorDirect(),
                ]
            )
            if s1_transform is None
            else s1_transform
        )

        self.s2_transform = (
            A.Compose(
                [
                    Sentinel2Normalize(),
                    ToTensorDirect(),
                ]
            )
            if s2_transform is None
            else s2_transform
        )

        self.lc_transform = (
            A.Compose(
                [
                    IGBP2DFCNormalize(channel_last=False),
                    ToTensorDirect(),
                ]
            )
            if lc_transform is None
            else lc_transform
        )

        self.output_packer = self._pack_output if output_packer is None else output_packer

    def prepare_data(self) -> None:
        super().prepare_data()

        def bar_progress(current, total, width=80):
            progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
            # Don't use print() as it will print in new line every time.
            sys.stdout.write("\r" + progress_message)
            sys.stdout.flush()

        for season in ["1158_spring", "1868_summer", "1970_fall", "2017_winter"]:
            for source in ["s1", "s2", "lc"]:
                fname = f"ROIs{season}"
                outname = os.path.join(self.root_dir, f"{fname}_{source}.tar.gz")

                if not os.path.exists(outname) and not os.path.exists(os.path.join(self.root_dir, fname)):
                    import wget

                    wget.download(
                        f"ftp://m1474000:m1474000@dataserv.ub.tum.de/{fname}.tar.gz", out=outname, bar=bar_progress
                    )

                if os.path.exists(outname) and not os.path.exists(os.path.join(self.root_dir, fname)):
                    import tarfile

                    with tarfile.open(outname) as file:
                        file.extractall(self.root_dir)

        for split in ["train", "test"]:
            fname = f"{split}_list.txt"
            outname = os.path.join(self.root_dir, fname)

            if not os.path.exists(outname):
                import wget

                wget.download(
                    f"https://raw.githubusercontent.com/schmitt-muc/SEN12MS/master/splits/{fname}",
                    out=outname,
                    bar=bar_progress,
                )

    def _pack_output(
        self,
        s1: Optional[TensorType] = None,
        s2: Optional[TensorType] = None,
        lc: Optional[TensorType] = None,
    ) -> Dict[str, Any]:
        sample = {}

        if isinstance(s1, TensorType) and isinstance(s2, TensorType):
            sample["image"] = torch.cat([s1, s2], dim=0)
        elif isinstance(s1, TensorType):
            sample["image"] = s1
        elif isinstance(s1, TensorType):
            sample["image"] = s2
        else:
            raise ValueError("No valid S1 or S2 Tensors were provided for packing")

        if lc is not None:
            sample["mask"] = lc

        return sample

    def preprocess(
        self,
        sample: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Transform a single sample from the Dataset.
        Args:
            sample: dictionary containing image and mask
        Returns:
            preprocessed sample
        """
        sample["image"] = sample["image"].astype(np.float32)

        split_idx = 2 if self.band_set in ["all", "s1"] else 0
        s1_img = sample["image"][:split_idx]
        s2_img = sample["image"][split_idx:]

        s1_img = self.s1_transform(image=s1_img)["image"] if s1_img.size > 0 else None
        s2_img = self.s2_transform(image=s2_img)["image"] if s2_img.size > 0 else None

        if "mask" in sample:
            lc_img = self.lc_transform(image=sample["mask"])["image"]
        else:
            lc_img = None

        return self.output_packer(s1=s1_img, s2=s2_img, lc=lc_img)

    def train_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for training.

        Returns:
            training data loader
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for validation.

        Returns:
            validation data loader
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for testing.

        Returns:
            testing data loader
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True,
        )


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    from represent.datamodules.represent_sen12ms_datamodule import *

    dm = RepresentSEN12MSDataModule("/Users/lloyd/Downloads/SEN12MS/Pretraining/SEN12MS", 42, batch_size=2)
    dm.prepare_data()
    dm.setup()

    ds = dm.train_dataset
    sample = ds[np.random.randint(0, len(ds))]
    sample["mask"] = sample["mask"][np.newaxis, :, :]
    ds.dataset.plot(sample)
    plt.show(block=True)

    # import code
    # import readline
    # import rlcompleter

    # vars = globals()
    # vars.update(locals())

    # readline.set_completer(rlcompleter.Completer(vars).complete)
    # readline.parse_and_bind("tab: complete")
    # code.InteractiveConsole(vars).interact()
