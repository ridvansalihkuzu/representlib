import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A

from typing import Union, Tuple, Text
from torchtyping import TensorType


class Sentinel1Normalize(A.ImageOnlyTransform):
    """Normalize a Sentinel-1 image tensor image by clipping its dB backscatter to the
    the Noise Equivilent Sigma Zero range or [0,99] percentile, and then shifting it to the range [0, 1]
    This transform does not support PIL Image.

    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.

    Args:
        toDb (bool, optional): Specify if the backscatter is in Decibels - no conversion is done if it is False.
        clip_to (string, optional): Clip to either the `nesz` (Noise Equivilent Sigma-0) or `percentile` to the [0,99]th percentile bounds
    """

    def __init__(
        self,
        toDb: bool = False,
        clip_to: Text = "nesz",
    ):
        super().__init__(True, 1)

        assert clip_to in ["percentile", "nesz"], "Valid options for `clip_to` are `nesz` or `percentile`"

        self.toDb = toDb
        self.clip_to = clip_to

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        """
        Args:
            tensor (Tensor): Tensor image to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        if self.clip_to == "nesz":
            if self.toDb:
                bmin = -25
                bmax = 0
            else:
                bmin = 10 ** (-25 / 10)
                bmax = 1
        elif self.clip_to == "percentile":
            bmin, bmax = np.quantile(img, [0.01, 0.99])

        clipped = img.astype(np.float32).clip(bmin, bmax)

        return (clipped - bmin) / (bmax - bmin)


class Sentinel2Normalize(A.ImageOnlyTransform):
    """Normalize a Sentinel-2 image tensor image by dividing its 16bit DN by the scale factor 10,000 and rescaling
    the tensor to the range [0,1]
    This transform does not support PIL Image.

    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.
    """

    def __init__(self):
        super().__init__(True, 1)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        """
        Args:
            tensor (Tensor): Tensor image to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        boa_reflectance = img.astype(np.float32).clip(0, 10000) / 10000
        return boa_reflectance


class IGBP2DFCNormalize(A.ImageOnlyTransform):
    IGBP2DFC = np.array([0, 1, 1, 1, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 6, 8, 9, 10])

    def __init__(self, channel_last: bool = True):
        super().__init__(True, 1)
        self.channel_last = channel_last

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        if self.channel_last:
            lc = img[:, :, 0].astype(np.int)
        else:
            lc = img[0, :, :].astype(np.int)
        lc = np.take(self.IGBP2DFC, lc)
        return lc
