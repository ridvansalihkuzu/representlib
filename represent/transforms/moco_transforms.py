import random
from matplotlib import image
import torch

import albumentations as A
import numpy as np

from typing import Any, Dict, Optional
from torchvision import transforms
from torchtyping import TensorType
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.augmentations.transforms import GaussianBlur

from represent.transforms.normalize import Sentinel1Normalize, Sentinel2Normalize, IGBP2DFCNormalize


# Because Augmentation libraries expect numpy ordering by TorchGeo Datasets provide Tensors.
def to_albumentations(img):
    if img is None:
        return None

    if len(img.shape) == 3:
        img = np.einsum("chw->hwc", img)

    if isinstance(img, torch.TensorType):
        return img.numpy()
    else:
        return img


class ToAlbumations(A.ImageOnlyTransform):
    def __init__(self):
        super().__init__(True, 1)

    def apply(self, img: np.ndarray, **params) -> torch.TensorType:
        return to_albumentations(img)


class ToTensorDirect(A.ImageOnlyTransform):
    def __init__(self):
        super().__init__(True, 1)

    def apply(self, img: np.ndarray, **params) -> torch.TensorType:
        return torch.from_numpy(img)


class NullTransform(A.ImageOnlyTransform):
    def __init__(self):
        super().__init__(True, 1)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return img


class AddSpeckle(A.ImageOnlyTransform):
    def __init__(
        self,
        ENL: int = 1,
        intensity: bool = True,
        cache_size: int = 100,
        isDb: bool = True,
        always_apply=False,
        p=1.0,
    ):
        """
        L -> ENL
        intensity = True use intensity image, False -> amplitude image
        """
        super().__init__(always_apply, p)

        self.L = ENL
        self.intensity = intensity
        self.isDb = isDb

        self.speckle_cache = []
        self.cache_size = cache_size

    def apply(self, img, **params):
        s = np.zeros(img.shape).astype(img.dtype)

        if len(self.speckle_cache) < self.cache_size:
            for k in range(self.L):
                s = s + np.abs(np.random.randn(*img.shape) + 1j * np.random.randn(*img.shape)) ** 2 / 2

            # s image de valeurs de pdf Gamma de nb de vues L de moyenne 1
            if self.intensity:
                s = s / self.L
            else:
                s = np.sqrt(s / self.L)

            s = s.astype(img.dtype)

            self.speckle_cache.append(s)
        else:
            idx = np.random.randint(0, self.cache_size)
            s = self.speckle_cache[idx]

        if self.isDb:
            img_lin = 10 ** (img / 10)
        else:
            img_lin = img

        ima_speckle_amplitude = img_lin * s

        if self.isDb:
            ima_speckle_amplitude = 10 * np.log10(ima_speckle_amplitude + 1e-6)

        return ima_speckle_amplitude.astype(img.dtype)


class MocoSEN12MSPacker:
    def __init__(
        self,
        bands: str = "all",
        patch_sz: int = 256,
        cache_size: int = 100,
    ) -> None:
        self.patch_sz = patch_sz
        self.bands = bands
        self.cache_size = cache_size

        larger_sz = min(256, int(patch_sz * 1.2))
        # These transforms are used to cut the patch from the larger image patch
        self.patch_selection = A.Compose(
            [
                A.RandomCrop(larger_sz, larger_sz, p=1),
            ],
            additional_targets={"s1": "image", "s2": "image"},
        )

        # These transforms transform the geospatial nature of the patch
        self.geo_transform = A.Compose(
            [
                A.RandomCrop(patch_sz, patch_sz, p=1),
                A.VerticalFlip(p=0.25),
                A.HorizontalFlip(p=0.25),
            ],
            additional_targets={"s1": "image", "s2": "image"},
        )

        # These transform the radiometric nature of the images
        self.s1_radio_transform = A.Compose(
            [
                AddSpeckle(ENL=4, intensity=False, isDb=True, p=0.5, cache_size=self.cache_size),
                Sentinel1Normalize(toDb=True, clip_to="nesz"),
                ToTensorV2(),
            ]
        )

        self.s2_radio_transform = A.Compose(
            [
                GaussianBlur(blur_limit=(3, 7), p=0.5),
                Sentinel2Normalize(),
                ToTensorV2(),
            ]
        )

        self.lc_transform = A.Compose(
            [
                IGBP2DFCNormalize(channel_last=True),
                ToTensorV2(),
            ]
        )

    def __call__(
        self,
        s1: Optional[TensorType] = None,
        s2: Optional[TensorType] = None,
        lc: Optional[TensorType] = None,
    ) -> Any:
        sample = {}

        # This image augmentation library is a pain to work with
        s1 = to_albumentations(s1)
        s2 = to_albumentations(s2)
        lc = to_albumentations(lc)

        images = {"image": lc}

        if self.bands in ["s1", "all"]:
            images["s1"] = s1

        if self.bands in ["all", "s2-all", "s2-reduced"]:
            images["s2"] = s2

        patches = self.patch_selection(**images)

        # Apply the geo transforms
        augmented_q = self.geo_transform(**patches)
        augmented_k = self.geo_transform(**patches)

        # Apply the modalit specific transforms
        if self.bands in ["s1", "all"]:
            augmented_q["s1"] = self.s1_radio_transform(image=augmented_q["s1"])["image"]
            augmented_k["s1"] = self.s1_radio_transform(image=augmented_k["s1"])["image"]

        if self.bands in ["all", "s2-all", "s2-reduced"]:
            augmented_q["s2"] = self.s2_radio_transform(image=augmented_q["s2"])["image"]
            augmented_k["s2"] = self.s2_radio_transform(image=augmented_k["s2"])["image"]

        augmented_q["image"] = self.lc_transform(image=augmented_q["image"])["image"]
        augmented_k["image"] = self.lc_transform(image=augmented_k["image"])["image"]

        if self.bands == "all":
            return (augmented_q["s1"], augmented_k["s1"], augmented_q["s2"], augmented_k["s2"]), augmented_q["image"]
        elif self.bands == "s1":
            return (augmented_q["s1"], augmented_k["s1"]), augmented_q["image"]
        else:
            return (augmented_q["s2"], augmented_k["s2"]), augmented_q["image"]
