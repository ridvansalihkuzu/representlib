from torchvision.transforms.functional import resize
from rasterio.windows import get_data_window
from itertools import product
from tqdm import tqdm

import os
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
import numpy as np
import copy

import torch

torch.manual_seed(0)
import random

random.seed(0)
import numpy as np

np.random.seed(0)

BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
NODATA_VALUE_LABELS = 255

CLASSIDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 25, 26]

# classes that are present in both train and test partitions
KEEP_CLASSES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 17, 18, 19, 20, 22, 23, 24]

# FOA LCCS classes
# classid, classid_coarse, coarse level (8 classes), fine level (26 classes)
CLASSES = [
    (1, 4, "Natural Waterbodies", "River"),
    (2, 2, "Cultivated and Managed Terrestrial Area(s)", "Tree Plantations Large"),
    (3, 1, "Natural vegetation", "Forest Closed"),
    (4, 1, "Natural vegetation", "Forest Open"),
    (5, 1, "Natural vegetation", "Shrubs Closed"),
    (6, 3, "Wetlands", "Grasslands Closed Acquatic"),
    (7, 5, "Urban", "Urban - Built Up"),
    (8, 1, "Natural vegetation", "Shrubs Open"),
    (9, 1, "Natural vegetation", "Grasslands Closed"),
    (10, 7, "Acquatic agriculture", "Agriculture flooded - Graminoid Small"),
    (11, 2, "Cultivated and Managed Terrestrial Area(s)", "Herbaceous Crops Small"),
    (12, 2, "Cultivated and Managed Terrestrial Area(s)", "Shrub Crops Small"),
    (13, 5, "Urban", "Urban - Not Built Up"),
    (14, 2, "Cultivated and Managed Terrestrial Area(s)", "Herbaceous Crops Large"),
    (15, 2, "Cultivated and Managed Terrestrial Area(s)", "Shrub Crops"),
    (16, 3, "Wetlands", "Shrubs Open Acquatic"),
    (17, 6, "Bare", "Bare Area(s)"),
    (18, 3, "Wetlands", "Shrubs Closed Acquatic"),
    (19, 1, "Natural Waterbodies", "Lake"),
    (20, 2, "Cultivated and Managed Terrestrial Area(s)", "Tree Orchard Small"),
    (21, 1, "Natural vegetation", "Grasslands Open"),
    (22, 2, "Cultivated and Managed Terrestrial Area(s)", "Tree Orchard Large"),
    (23, 3, "Wetlands", "Forest Open Acquatic"),
    (24, 3, "Wetlands", "Forest Closed Acquatic"),
    (25, 2, "Cultivated and Managed Terrestrial Area(s)", "Tree Plantations Small"),
    (26, 7, "Acquatic agriculture", "Agriculture flooded - Graminoid Large"),
]

CLASSNAMES = {classid: name for classid, classid_coarse, name_coarse, name in CLASSES}
classid_detailed_to_coarse_mapping = {
    classid: classid_coarse for classid, classid_coarse, name_coarse, name in CLASSES
}

"categories Table 1 RepreSent-e-Geos-RPT D1.1 20220627 (page 21)"
COARSE_CLASSES = {
    1: "Natural Vegetation",
    2: "Cultivated and Managed Terrestrial Area(s)",
    3: "Wetlands",
    4: "Natural Waterbodies",
    5: "Urban",
    6: "Bare",
    7: "Acquatic agriculture",
}

TRAIN_TILES = [
    "37MBM",
    "37MBN",
    "37MCM",
    "37MCN",
    "37MDN",
    "36LZL",
    "36LZM",
    "36LZN",
    "36LZP",
    "36LZQ",
    "36LZR",
    "36MZS",
    "37LBF",
    "37LBG",
    "37LBJ",
    "37LBK",
    "37LCF",
    "37LCG",
    "37LCH",
    "37LCJ",
    "37LCK",
    "37LCL",
    "37LDG",
    "37LDH",
    "37LDK",
    "37LDL",
]

TEST_TILES = ["37LBH", "37LBL", "37MDM"]


class UC2LandCoverDataset(Dataset):
    def __init__(
        self,
        root,
        partition,
        segmentation=False,
        keep_classes=None, # hard-coded at the moment
        simplified_classes=True,
        keep_tiles=None,
        image_size=2240,
        transform=None,
        dataset_fraction=1.0,
    ):
        """
        root: folder to dataset. Must contain Train and Test folders with tile/date/B*.jp2 scenes
        partition: "Train" or "Test"
        segmentation: if True, returns a segmentation map, if False a single label per patch (most common class)
        keep_classes: list if not None, drops all classes that are not in this list
        image_size tile size in meters
        """

        self.root = root
        self.partition = partition
        self.segmentation = segmentation
        self.simplified_classes = simplified_classes
        self.image_size = image_size
        self.transform = transform
        self.keep_classes = keep_classes
        self.dataset_fraction = dataset_fraction

        indexfile = os.path.join(root, f"patches_{self.image_size}", f"{partition}_index.csv")
        if not os.path.exists(indexfile):
            print(f"building indexfile {indexfile}")
            df = self.build_index()
            df.to_csv(indexfile)
        self.index = pd.read_csv(indexfile, index_col=0)

        # add coarse classids to index
        self.index["majorityclass_coarse"] = [classid_detailed_to_coarse_mapping[c] for c in self.index.majorityclass]

        if keep_classes is not None:
            self.index = self.index.loc[self.index["majorityclass"].isin(keep_classes)]

        if keep_tiles is not None:
            self.index = self.index.loc[self.index["tile"].isin(keep_tiles)]

        if self.dataset_fraction is not None and self.dataset_fraction < 1.0:
            self.index = self.index.groupby("majorityclass", group_keys=False).apply(
                lambda x: x.sample(frac=self.dataset_fraction)
            )

    def get_class_weights(self):
        index = pd.concat([self.index])
        classes, counts = np.unique(index.majorityclass, return_counts=True)
        class_weights = counts.sum() / (len(classes) * counts)
        class_weights /= class_weights.min()
        return class_weights

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):

        row = self.index.iloc[idx]

        s2_path = os.path.join(
            self.root, f"patches_{self.image_size}", self.partition, f"{row.tile}_{row.date}_{row.patch:0>4}_s2.tif"
        )
        lc_path = os.path.join(
            self.root, f"patches_{self.image_size}", self.partition, f"{row.tile}_{row.date}_{row.patch:0>4}_lc.tif"
        )

        with rasterio.open(s2_path, "r") as src:
            s2 = src.read()
        s2 = torch.from_numpy(s2 * 1e-4).float()

        if self.segmentation:
            with rasterio.open(lc_path, "r") as src:
                lc = src.read(1)

                # replace classids of image with simplified coarse class ids
                if self.simplified_classes:
                    lc_coarse = copy.copy(lc)
                    for k, v in classid_detailed_to_coarse_mapping.items():
                        lc_coarse[lc == k] = v
                    lc = lc_coarse
        else:
            labelcolumn = "majorityclass_coarse" if self.simplified_classes else "majorityclass"
            lc = np.array(row[labelcolumn])

            # convert class id into index of keep_classes list
            lc = list(self.keep_classes).index(lc)

        lc = torch.tensor(lc).long()

        if self.transform is not None:
            s2 = self.transform(s2.numpy())

        return s2, lc

    def build_index(self):
        patch_folder = os.path.join(self.root, f"patches_{self.image_size}", self.partition)

        scenes = [f for f in os.listdir(patch_folder) if f.endswith("s2.tif")]

        stats = []
        for scene in tqdm(scenes, desc="building indexfile"):
            tile, date, patch, *_ = scene.split("_")

            with rasterio.open(os.path.join(patch_folder, scene.replace("s2", "lc")), "r") as src:
                labels = src.read()
                left, bottom, right, top = src.bounds
                crs = str(src.crs)

            classids, counts = np.unique(labels, return_counts=True)

            p = counts / counts.sum()
            entropy = (-p * np.log(p)).sum()

            stat = dict(
                tile=tile,
                date=date,
                patch=patch,
                left=left,
                bottom=bottom,
                right=right,
                top=top,
                crs=crs,
                entropy=entropy,
                majorityclass=classids[counts.argmax()],
            )
            for classid, count in zip(classids, counts):
                stat[f"class{classid}"] = count

            stats.append(stat)

        df = pd.DataFrame(stats).fillna(0)

        return df


from argparse import ArgumentParser


class UC2LandCoverDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "/data/RepreSent",
        batch_size: int = 64,
        segmentation=False,
        simplified_classes=False,
        keep_classes=None,
        image_size=2240,
        num_workers=8,
        cropsize=48,
        transform=None,
        num_val_tiles=5,
        training_set_fraction=1.0,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.segmentation = segmentation
        self.simplified_classes = simplified_classes
        self.keep_classes = keep_classes
        self.image_size = image_size
        self.num_workers = num_workers
        self.cropsize = cropsize
        self.transform = transform
        self.num_val_tiles = num_val_tiles
        self.training_set_fraction = training_set_fraction

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--data_dir", type=str, default="/data/RepreSent/UC2")
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--num_val_tiles", type=int, default=4)
        parser.add_argument("--cropsize", type=int, default=48)
        parser.add_argument("--segmentation", action="store_true")
        parser.add_argument("--simplified_classes", action="store_true")
        parser.add_argument("--image_size", type=int, default=610)
        parser.add_argument("--training_set_fraction", type=float, default=1)
        return parser

    def prepare_data(self) -> None:
        if "Test" not in os.listdir(self.data_dir) or "Train" not in os.listdir(self.data_dir):
            print(f"could not find Train or Test folders in {self.data_dir}. please download data")
            return

        if f"patches_{self.image_size}" not in os.listdir(self.data_dir):
            print(f"could not find patches_{self.image_size} folder in {self.data_dir}. Creating patches...")
            create_patches(self.data_dir, self.image_size)

    def setup(self, stage="fit"):

        if stage == "fit" or stage is None:
            val_tiles = np.random.choice(TRAIN_TILES, self.num_val_tiles, replace=False)
            train_tiles = [t for t in TRAIN_TILES if t not in val_tiles]

            print(f"training tiles: {train_tiles}")
            print(f"validation tiles: {val_tiles}")

            self.train_dataset = UC2LandCoverDataset(
                root=self.data_dir,
                partition="Train",
                segmentation=self.segmentation,
                simplified_classes=self.simplified_classes,
                keep_classes=self.keep_classes,
                image_size=self.image_size,
                transform=self.transform,
                keep_tiles=train_tiles,
                dataset_fraction=self.training_set_fraction,
            )

            self.val_dataset = UC2LandCoverDataset(
                root=self.data_dir,
                partition="Train",
                segmentation=self.segmentation,
                simplified_classes=self.simplified_classes,
                keep_classes=self.keep_classes,
                image_size=self.image_size,
                transform=None,  # no transform on the validation dataset
                keep_tiles=val_tiles,
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_dataset = UC2LandCoverDataset(
                root=self.data_dir,
                partition="Test",
                segmentation=self.segmentation,
                simplified_classes=self.simplified_classes,
                keep_classes=self.keep_classes,
                image_size=self.image_size,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


def load_s2_image(imagepath, bounds, expected_image_size):
    left, bottom, right, top = bounds

    # extract bands and image sizes
    band_stack = []
    for band in BANDS:
        with rasterio.open(os.path.join(imagepath, band + ".jp2")) as src:
            patch_window = rasterio.windows.from_bounds(left, bottom, right, top, src.transform)
            band_stack.append(src.read(1, window=patch_window))

    # extract dimensions of the B02.jpg band (10m) to rescale other images to this size
    height, width = band_stack[1].shape

    size_in_px = expected_image_size // 10
    if height == size_in_px and width == size_in_px:

        # bilinearly interpolate all bands to 10m using torch functional resize
        band_stack = [
            resize(torch.from_numpy(b[None].astype("int32")), [height, width]).squeeze(0).numpy() for b in band_stack
        ]

        # stack to image [13 x H x W]
        image = np.stack(band_stack)

        # valid pixels are where there is no 0 in all bands
        invalid_mask = image.sum(0) == 0

        if not np.isnan(image).any():

            if not invalid_mask.any():

                # prepare metadata for storing a georeferenced patch on disk

                # extract transform of the window from a 10m band
                with rasterio.open(os.path.join(imagepath, BANDS[1] + ".jp2")) as src:
                    patch_window = rasterio.windows.from_bounds(left, bottom, right, top, src.transform)
                    win_transform = src.window_transform(patch_window)
                    profile = src.profile
                    profile["width"], profile["height"], profile["count"] = width, height, len(BANDS)
                    profile["transform"] = win_transform

                return image, profile

    # if image not in correct size or contains invalid data
    return None, None


def read_label(path, bounds):
    left, bottom, right, top = bounds

    with rasterio.open(path) as src:
        patch_window = rasterio.windows.from_bounds(left, bottom, right, top, src.transform)
        labels = src.read(1, window=patch_window)
        labels = np.nan_to_num(labels).astype(int)
        win_transform = src.window_transform(patch_window)
        profile = src.profile
        profile["width"], profile["height"] = labels.shape
        profile["transform"] = win_transform
        profile["dtype"] = "uint16"
        profile["nodata"] = NODATA_VALUE_LABELS

    return labels, profile


def create_patches(data_dir, image_size):

    for partition in ["Test", "Train"]:
        print(f"partition {partition}")
        tile_path = os.path.join(data_dir, partition)
        tiles = os.listdir(tile_path)

        for i, tile in enumerate(tiles):
            print(f"tile {tile} ({i}/{len(tiles)})")
            s2_dates = os.listdir(os.path.join(tile_path, tile))  # returns ['20170625', '20170723', 'labels']
            s2_dates.remove("labels")  # removes labels from the list

            with rasterio.open(os.path.join(tile_path, tile, "labels", "raster", tile + ".tif")) as src:
                left, bottom, right, top = src.bounds

            # define a regular patch-grid of SIZExSIZE
            lr = np.arange(left, right, image_size)
            bt = np.arange(bottom, top, image_size)

            # generate bottom-left coordinates of the patches
            patches = list(product(lr, bt))

            for s2_date in s2_dates:
                for patch_idx, patch in tqdm(
                    enumerate(patches),
                    total=len(patches),
                    leave=False,
                    desc=f"creating {image_size}m x {image_size}m patches from tile {tile}-{s2_date}",
                ):

                    left, bottom, right, top = patch[0], patch[1], patch[0] + image_size, patch[1] + image_size
                    bounds = left, bottom, right, top

                    # read s2
                    image, profile = load_s2_image(
                        os.path.join(tile_path, tile, s2_date), bounds, expected_image_size=image_size
                    )

                    # write tile
                    if image is not None:
                        s2_write_path = (
                            f"{data_dir}/patches_{image_size}/{partition}/{tile}_{s2_date}_{patch_idx:0>4}_s2.tif"
                        )
                        lc_write_path = (
                            f"{data_dir}/patches_{image_size}/{partition}/{tile}_{s2_date}_{patch_idx:0>4}_lc.tif"
                        )

                        # write s2
                        os.makedirs(os.path.dirname(s2_write_path), exist_ok=True)
                        with rasterio.open(s2_write_path, "w", **profile) as dst:
                            dst.write(image.astype(rasterio.uint16))

                        # read label
                        labels, profile = read_label(
                            os.path.join(tile_path, tile, "labels", "raster", tile + ".tif"), bounds
                        )

                        # write label
                        with rasterio.open(lc_write_path, "w", **profile) as dst:
                            dst.write(labels[None])


"""Meta learning wrappers"""


def split_support_query(ds, shots, at_least_n_queries=1, random_state=0, classcolumn="majorityclass"):
    classes, counts = np.unique(ds.index[classcolumn], return_counts=True)
    classes = classes[
        counts > (shots + at_least_n_queries)
    ]  # we need at least shots + 1 + at_least_n_queries samples of each class in the dataset

    supports = []
    queries = []
    for c in classes:
        samples = ds.index.loc[ds.index[classcolumn] == c].reset_index()
        support = samples.sample(shots, random_state=random_state)
        query = samples.drop(support.index)
        supports.append(support)
        queries.append(query)
    supports = pd.concat(supports)
    queries = pd.concat(queries)

    return supports, queries


def load_samples(ds, dataframe):
    data_input, data_target = [], []
    for idx in dataframe["index"]:
        row = ds.index.loc[idx]
        i = (ds.index.index == idx).argmax()
        im, label = ds[i]
        data_input.append(im)
        data_target.append(label)

    data_input = torch.stack(data_input)
    data_target = torch.tensor(data_target)
    return data_input, data_target


def sample_task(ds, shots, ways):

    index = ds.index.reset_index()
    tiles = index.tile.unique()

    tile = tiles[0]
    index = index.loc[index.tile == tile]
    classes, counts = np.unique(index.majorityclass, return_counts=True)

    # drop classes with too few data
    classes = classes[counts > 2 * shots]

    # select shot classes
    classes = np.random.choice(classes, ways, replace=False)

    supports, queries = [], []
    for c in classes:
        classindex = index.loc[index.majorityclass == c]
        support = classindex.sample(shots)
        supports.append(support)
        queries.append(classindex.loc[~classindex.index.isin(support.index)].sample(shots))
    supports = pd.concat(supports)
    queries = pd.concat(queries)

    X_support, y_support = load_samples(ds, supports)
    X_query, y_query = load_samples(ds, queries)
    return X_support, y_support, X_query, y_query


class UC2LandCoverTaskDataset(Dataset):
    def __init__(
        self,
        root,
        partition,
        segmentation=False,
        simplified_classes=True,
        image_size=2240,
        keep_tiles=None,
        transform=None,
        episodes_per_epoch=1000,
        shots=2,
        ways=4,
    ):
        self.ds = UC2LandCoverDataset(
            root=root,
            partition=partition,
            segmentation=segmentation,
            keep_tiles=keep_tiles,
            transform=transform,
            simplified_classes=simplified_classes,
            image_size=image_size,
        )
        self.episodes_per_epoch = episodes_per_epoch
        self.shots = shots
        self.ways = ways

    def __len__(self):
        return self.episodes_per_epoch

    def __getitem__(self, _):
        return sample_task(self.ds, self.shots, self.ways)


class UC2LandCoverTaskDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "/data/RepreSent",
        batch_size: int = 64,
        segmentation=False,
        simplified_classes=False,
        image_size=2240,
        num_workers=8,
        cropsize=48,
        transform=None,
        num_val_tiles=4,
        episodes_per_epoch=1000,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.segmentation = segmentation
        self.simplified_classes = simplified_classes
        self.image_size = image_size
        self.num_workers = num_workers
        self.cropsize = cropsize
        self.transform = transform
        self.num_val_tiles = num_val_tiles
        self.episodes_per_epoch = episodes_per_epoch

    def prepare_data(self) -> None:
        if "Test" not in os.listdir(self.data_dir) or "Train" not in os.listdir(self.data_dir):
            print(f"could not find Train or Test folders in {self.data_dir}. please download data")
            return

        if f"patches_{self.image_size}" not in os.listdir(self.data_dir):
            print(f"could not find patches_{self.image_size} folder in {self.data_dir}. Creating patches...")
            create_patches(self.data_dir, self.image_size)

    def setup(self, stage="fit"):

        if stage == "fit" or stage is None:
            val_tiles = np.random.choice(TRAIN_TILES, self.num_val_tiles, replace=False)
            train_tiles = [t for t in TRAIN_TILES if t not in val_tiles]

            print(f"training tiles: {train_tiles}")
            print(f"validation tiles: {val_tiles}")

            self.train_dataset = UC2LandCoverTaskDataset(
                root=self.data_dir,
                partition="Train",
                segmentation=self.segmentation,
                simplified_classes=self.simplified_classes,
                image_size=self.image_size,
                transform=self.transform,
                keep_tiles=train_tiles,
                episodes_per_epoch=self.episodes_per_epoch,
            )

            self.val_dataset = UC2LandCoverTaskDataset(
                root=self.data_dir,
                partition="Train",
                segmentation=self.segmentation,
                simplified_classes=self.simplified_classes,
                image_size=self.image_size,
                transform=None,  # no augmentations on the validation set
                keep_tiles=val_tiles,
                episodes_per_epoch=self.episodes_per_epoch,
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            raise NotImplementedError()

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


if __name__ == "__main__":
    data_dir = "/data/RepreSent/UC2"
    dm = UC2LandCoverDataModule(data_dir, batch_size=64, segmentation=True, simplified_classes=False, image_size=2440)
    dm.prepare_data()
    dm.setup()
    train_ds = dm.train_dataset

    from skimage.exposure import equalize_hist
    import matplotlib.pyplot as plt

    def s2_to_rgb(image):
        return equalize_hist(image[np.array([3, 2, 1])]).transpose(1, 2, 0)

    idxs = np.random.randint(len(train_ds), size=10)
    # idxs = np.array([ 21496]) # np.random.randint(len(train_ds), size=10)

    for idx in idxs:
        image, labels = train_ds[idx]

        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(s2_to_rgb(image.numpy()))
        axs[1].imshow(labels, vmin=0, vmax=26, cmap="hsv")
        [ax.axis("off") for ax in axs]

        ids, counts = np.unique(labels, return_counts=True)
        plt.suptitle(f"idx {idx}; class {ids[0]}")
        plt.tight_layout()

    plt.show()

    # Meta-Learning tasks
    shots = 2
    ways = 5

    task_ds = UC2LandCoverTaskDataset(
        root=data_dir,
        partition="Test",
        segmentation=False,
        simplified_classes=False,
        image_size=610,
        shots=shots,
        ways=ways,
    )

    X_support, y_support, X_query, y_query = task_ds[0]

    fig, axs = plt.subplots(shots, ways, figsize=(ways * 3, shots * 3), sharex=True)
    fig.suptitle("support (train)")
    for i, (ax, X, y) in enumerate(zip(axs.T.reshape(-1), X_support, y_support)):
        ax.imshow(s2_to_rgb(X.numpy()))
        ax.set_xticks([])
        ax.set_yticks([])
        if i % shots == 0:
            ax.set_title(CLASSNAMES[int(y)])

    plt.tight_layout()

    fig, axs = plt.subplots(shots, ways, figsize=(ways * 3, shots * 3), sharex=True)
    fig.suptitle("query (test)")
    for i, (ax, X, y) in enumerate(zip(axs.T.reshape(-1), X_query, y_query)):
        ax.imshow(s2_to_rgb(X.numpy()))
        ax.set_xticks([])
        ax.set_yticks([])
        if i % shots == 0:
            ax.set_title(CLASSNAMES[int(y)])

    plt.tight_layout()

    plt.show()
