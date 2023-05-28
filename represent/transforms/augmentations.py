"""Transforms"""

import numpy as np
import torch
import random

def get_data_augmentation(cropsize):
    """
    do data augmentation:
    model
    """
    def data_augmentation(image, mask=None):
        segmentation = mask is not None

        image = torch.Tensor(image)
        if segmentation:
            mask = torch.Tensor(mask)
            mask = mask.unsqueeze(0)

        rot = np.random.choice([0, 1, 2, 3])
        image = torch.rot90(image, rot, [1, 2])
        if segmentation:
            mask = torch.rot90(mask, rot, [1, 2])

        if random.random() < 0.5:
            # flip left right
            image = torch.fliplr(image.permute(1,2,0)).permute(2,0,1)
            if segmentation:
                mask = torch.fliplr(mask.permute(1,2,0)).permute(2,0,1)

        if random.random() < 0.5:
            # flip up-down
            image = torch.flipud(image.permute(1,2,0)).permute(2,0,1)
            if segmentation:
                mask = torch.flipud(mask.permute(1,2,0)).permute(2,0,1)

        # a slight rescaling
        scale_factor = np.random.normal(1, 1e-1)
        min_scale_factor = (cropsize + 5) / image.shape[1] # clamp scale factor so that random crop to certain cropsize is still possible
        scale_factor = np.max([scale_factor, min_scale_factor])

        image = torch.nn.functional.interpolate(image.unsqueeze(0), scale_factor=scale_factor, mode="nearest", recompute_scale_factor=True).squeeze(0)
        if segmentation:
            mask = torch.nn.functional.interpolate(mask.unsqueeze(0), scale_factor=scale_factor, mode="nearest", recompute_scale_factor=True).squeeze(0)

        if segmentation:
            image, mask = random_crop(image, mask, cropsize=cropsize)
        else:
            image = random_crop(image, cropsize=cropsize)

        std_noise = 0.2 * image.std()
        if random.random() < 0.5:
            # add noise per pixel and per channel
            pixel_noise = torch.rand(image.shape[1], image.shape[2])
            pixel_noise = torch.repeat_interleave(pixel_noise.unsqueeze(0), image.size(0), dim=0)
            image = image + pixel_noise*std_noise

        if random.random() < 0.5:
            channel_noise = torch.rand(image.shape[0]).unsqueeze(1).unsqueeze(2)
            channel_noise = torch.repeat_interleave(torch.repeat_interleave(channel_noise, image.shape[1], 1),
                                                    image.shape[2], 2)
            image = image + channel_noise*std_noise

            if random.random() < 0.5:
                # add noise
                noise = torch.rand(image.shape[0], image.shape[1], image.shape[2]) * std_noise
                image = image + noise

        if segmentation:
            mask = mask.squeeze(0)
            return image, mask
        else:
            return image

    return data_augmentation

def random_crop(image, cropsize, mask=None):
    C, W, H = image.shape
    w, h = cropsize, cropsize

    # distance from image border
    dh, dw = h // 2, w // 2

    # sample some point inside the valid square
    x = np.random.randint(dw, W - dw)
    y = np.random.randint(dh, H - dh)

    # crop image
    image = image[:, x - dw:x + dw, y - dh:y + dh]
    if mask is not None:
        mask = mask[:, x - dw:x + dw, y - dh:y + dh]

        return image, mask
    else:
        return image
