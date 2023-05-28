import torch
import rasterio as rio
import pandas as pd
import geopandas as gpd
import os
from skimage.morphology import binary_opening
import numpy as np
from rasterio import features
from shapely.geometry import shape
from shapely.geometry import Polygon

from itertools import product
import numpy as np
import geopandas as gpd
from torch.utils.data import Dataset
from glob import glob
from torchvision.transforms import Resize
import torchvision
import rasterio
from torchvision.transforms.functional import resize
import numpy as np
from scipy.stats import skewcauchy
from tqdm import tqdm
from copy import copy

BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]

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

class SettlementDataset(Dataset):
    def __init__(self, data_path,
                 tile,
                 imagesize=640, # imagesize in meter
                 segmentation=False,
                 overwrite=False
                 ):
        # prepare data (is skipped if already present)
        mosaik(data_path=data_path, tile=tile, overwrite=overwrite)
        write_urban_tif_and_shape(data_path, tile, overwrite=overwrite)

        # initialization
        self.index = gpd.read_file(os.path.join(data_path, "Test", tile, "labels", "urban", f"{tile}.shp"), index_col=0)
        self.imagepath = os.path.join(data_path, "Test", tile, "mosaik")
        self.imagesize = imagesize
        self.segmentation = segmentation

        if segmentation:
            gdf = gpd.read_file(os.path.join(data_path, "Test", tile, "labels", "vector", f"{tile}.shp"))
            self.shapes = gdf.loc[gdf["a_name"] == "Urban"]

        print("checking samples...")
        valid = [self[i] is not None for i in tqdm(range(len(self)))]
        self.index = self.index.loc[np.array(valid)]
        print(f"dropping {(~np.array(valid)).sum()} invalid samples")


    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        geometry = self.index.iloc[item].geometry
        x,y = geometry.centroid.x, geometry.centroid.y

        left, bottom, right, top = x, y, x + self.imagesize, y + self.imagesize
        bounds = left, bottom, right, top

        s2, meta = load_s2_image(self.imagepath, bounds, expected_image_size=self.imagesize)

        # stop early if loading failed (is checked for "checking samples above")
        if s2 is None:
            return None

        if self.segmentation:
            targets = rio.features.rasterize(self.shapes.geometry, all_touched=True,
                                      transform=meta["transform"], out_shape=s2[0].shape)
            return s2, targets, meta
        else:
            return s2, meta
        
    
    def get_image(self, cx, cy, imagesize):

        left, bottom, right, top = cx - imagesize // 2, cy - imagesize // 2, cx + imagesize // 2, cy + imagesize // 2
        bounds = left, bottom, right, top

        s2, meta = load_s2_image(self.imagepath, bounds, expected_image_size=imagesize)

        targets = rio.features.rasterize(self.shapes.geometry, all_touched=True,
                              transform=meta["transform"], out_shape=s2[0].shape)

        return s2, targets, meta
        
def get_center(m):
    pixel_size = m["transform"].a
    x = m["transform"].c
    y = m["transform"].f

    cx = x + m["width"] // 2 * pixel_size
    cy = y - m["height"] // 2 * pixel_size
    return cx, cy


def make_grid(polygon, edge_size):
    """
    polygon : shapely.geometry
    edge_size : length of the grid cell
    from https://stackoverflow.com/questions/68770508/st-make-grid-method-equivalent-in-python/68778560#68778560
    """
    bounds = polygon.bounds
    x_coords = np.arange(bounds[0] + edge_size / 2, bounds[2], edge_size)
    y_coords = np.arange(bounds[1] + edge_size / 2, bounds[3], edge_size)
    combinations = np.array(list(product(x_coords, y_coords)))
    squares = gpd.points_from_xy(combinations[:, 0], combinations[:, 1]).buffer(edge_size / 2, cap_style=3)
    return gpd.GeoSeries(squares[squares.intersects(polygon)])

def write_urban_tif_and_shape(data_path, tile, urban_class=7,
                              erosion_size=3, overwrite=False,
                              simplify_radius=10, edge_size = 640):
    """
    creates a raster file of urban areas only. these areas are eroded by a certain amount
    creates a shapefile of settlements by vectoriting the eroded raster file
    """

    labels_tif = os.path.join(data_path, "Test", tile, "labels", "raster", f"{tile}.tif")
    target_tif = os.path.join(data_path, "Test", tile, "labels", "urban", f"{tile}.tif")
    target_shp = os.path.join(data_path, "Test", tile, "labels", "urban", f"{tile}.shp")

    os.makedirs(os.path.dirname(target_tif), exist_ok=True)

    if os.path.exists(target_tif) and os.path.exists(target_shp) and not overwrite:
        print(f"files {target_tif} and {target_shp} exist. skipping, specificy overwrite=True to rewrite")
        return

    with rio.open(labels_tif, "r") as src:
        lab = (src.read(1) == urban_class)
        lab = binary_opening(lab, footprint=np.ones((erosion_size, erosion_size)))
        lab = np.nan_to_num(lab, nan=255)

        profile = src.profile
        profile.update(
            dtype="uint8",
            nodata="255"
        )

        with rio.open(target_tif, "w", **profile) as dst:
            dst.write(lab, 1)
        print(f"wrote {target_tif}")

        shapes = features.shapes((lab == 1).astype("uint8"), transform=src.transform)

        geoms = [Polygon(record["coordinates"][0]) for (record, i) in shapes]
        gdf = gpd.GeoDataFrame(geometry=geoms, crs=profile["crs"])
        b = gdf.iloc[-1]
        boundary = gpd.GeoDataFrame([1], geometry=[b.geometry], crs=profile["crs"])
        gdf = gdf.iloc[:-2]# drop last row, as it is a polygon of the entire image
        gdf = gdf.dissolve().explode(index_parts=True)
        if simplify_radius > 0:
            gdf.geometry = gdf.geometry.simplify(simplify_radius) # simplify at 10m resolution to avoid pixel corners

        # split large polygons into smaller ones
        geometries = []
        for (_, idx), row in gdf.iterrows():
            if row.geometry.area > edge_size**2:
                geoms = make_grid(row.geometry, edge_size=edge_size)
                geoms = gpd.GeoSeries(geoms, crs=profile["crs"])
                split_idx = list(geoms.index)
                geoms.index = [f"{idx}-{i}" for i in split_idx]
                geometries.append(geoms)
            else:
                _, idx = row.name
                series = gpd.GeoSeries(row, crs=profile["crs"])
                series.index = [f"{idx}-0"]
                geometries.append(series)
        blocks = pd.concat(geometries)

        gdf = gpd.clip(blocks, gdf, keep_geom_type=True)
        #msk = (gdf.geometry.type == "Polygon") | (gdf.geometry.type == "MultiPolygon")
        #multipolys = gdf.loc[]
        #gdf = gdf.loc[msk] # removing GeometryCollections

        gdf.to_file(target_shp)
        print(f"wrote {target_shp}")

def mosaik(data_path, tile, overwrite=False):
    target_path = os.path.join(data_path, "Test", tile, "mosaik")
    os.makedirs(target_path, exist_ok=True)
    bands = glob(os.path.join(data_path, "Test", tile, "*", "B*.jp2"))
    scene = [b.split("/")[-2] for b in bands]
    b = [b.split("/")[-1].replace(".jp2", "") for b in bands]
    df = pd.DataFrame([bands, scene, b], index=["path", "scene", "band"]).T
    for band in BANDS:
        trg_file = os.path.join(target_path, f"{band}.jp2")
        if os.path.exists(trg_file) and not overwrite:
            print(f"{trg_file} exists. skipping. specify overwrite=True to regenerate mosaik")
            continue

        df_ = df.loc[df.band == band]

        arrs = []
        for idx, row in df_.iterrows():
            with rio.open(row.path, "r") as src:
                arrs.append(src.read(1))
                profile = src.profile

        arrs = np.stack(arrs).astype("float16")
        arrs[arrs == 0] = np.nan
        mosaik = np.nan_to_num(np.nanmin(arrs,axis=0)).astype("uint16")

        with rio.open(trg_file, "w", **profile) as dst:
            dst.write(mosaik, 1)
            print(f"writing {trg_file}")

def sample_settlements(data_source,
                   target_index,
                   num_samples=50,
                   dist_rv=skewcauchy(a=0.999, loc=5000, scale=10000),
                   return_idx_p=False,
                   seed=0):
    # makes sure p sums to one
    def normalize(x):
        return (x / x.sum())

    target_sample = data_source.index.iloc[target_index]
    geom = target_sample.geometry
    x, y = geom.centroid.x, geom.centroid.y

    distances = []
    for idx, row in data_source.index.iterrows():
        c = row.geometry.centroid
        cx, cy = c.x, c.y
        distances.append(np.sqrt((x - cx) ** 2 + (y - cy) ** 2))
    distances = np.array(distances)

    p = normalize(dist_rv.pdf(distances))

    idxs = np.random.RandomState(seed).choice(np.arange(len(data_source)), replace=False, p=p, size=num_samples)

    batch = np.stack([data_source[idx] for idx in idxs])



    if data_source.segmentation:
        X,y, meta = map(list, zip(*batch))
        batch = (np.stack(X), np.stack(y), meta)
    else:
        batch = np.stack(batch)

    if return_idx_p:
        return batch, idxs, p
    else:
        return batch
    
def sample_negatives(data_source,
                   target_index,
                   num_samples=50,
                   dist_rv=skewcauchy(a=0.999, loc=5000, scale=10000),
                   seed=0):
    # makes sure p sums to one
    segmentation = data_source.segmentation

    imagesize = data_source.imagesize
    imagepath = data_source.imagepath

    target_sample = data_source.index.iloc[target_index]
    geom = target_sample.geometry
    cx, cy = geom.centroid.x, geom.centroid.y

    # sample polar coordinates
    n_coordinates = num_samples*4 # sample more coordinates in case some are invalid
    distances = dist_rv.rvs(n_coordinates, random_state=seed)
    angles = np.random.RandomState(seed).randn(n_coordinates) * 2 * np.pi

    X = cx + distances * np.cos(angles)
    Y = cy + distances * np.sin(angles)

    batch = []
    targets = []
    metas = []
    for x,y in zip(X,Y):
        left, bottom, right, top = x, y, x + imagesize, y + imagesize
        bounds = left, bottom, right, top

        s2, meta = load_s2_image(imagepath, bounds, expected_image_size=imagesize)
        if s2 is not None: # if valid

            if segmentation:
                t = rio.features.rasterize(data_source.shapes.geometry, all_touched=True,
                                          transform=meta["transform"], out_shape=s2[0].shape)
                targets.append(t)

            batch.append(s2)
            metas.append(meta)

        # stop early if sufficient valid samples have been found
        if len(batch) >= num_samples:
            break

    if segmentation:
        return np.stack(batch), np.stack(targets), metas
    else:
        return np.stack(batch), metas

def sample_batch(data_source, target_index, num_shots=10, dist_rv=skewcauchy(a=0.999, loc=5000, scale=10000)):
    num_shots_pos, num_shots_neg = num_shots
    
    pos_batch = sample_settlements(data_source, target_index=target_index, num_samples=num_shots_pos, dist_rv=dist_rv)
    neg_batch = sample_negatives(data_source, target_index=target_index, num_samples=num_shots_neg, dist_rv=dist_rv)

    if not data_source.segmentation:
        pos_target = np.ones(pos_batch.shape[0], dtype=int)
        neg_target = np.zeros(pos_batch.shape[0], dtype=int)

        return np.vstack([pos_batch, neg_batch]), np.hstack([pos_target, neg_target])

    else:
        pos_X, pos_target, pos_meta = pos_batch
        neg_X, neg_target, neg_meta = neg_batch

        return np.vstack([pos_X, neg_X]), np.vstack([pos_target, neg_target]), pos_meta + neg_meta


def load_uc2_settlement_data(
        num_shots=(200, 600),
        imagesize=10240,
        target_index=600,
        datapath="/data/RepreSent/UC2",
        savepath=None,
        use_cache=False):  # f"/home/marc/Desktop/uc2_settlements/{target_index}"
    
    if use_cache and savepath is not None:
        data = torch.load(os.path.join(savepath, 'data.npz'))
        return data["X"], data["Y"], data["x_test"], data["y_test"], data["buildings"], data["meta"]

    ds = SettlementDataset(datapath, "37LBL", segmentation=True)

    x, y, meta_ = ds[target_index]
    cx, cy = get_center(meta_)

    x_test, y_test, meta = ds.get_image(cx, cy, imagesize=imagesize)

    labelprofile = copy(meta)
    labelprofile.update(
        count=1)

    dist = skewcauchy(a=0.999, loc=7500, scale=10000)
    print("sampling batch")
    X, Y, train_metas = sample_batch(ds, target_index=target_index, num_shots=num_shots, dist_rv=dist)

    X = torch.from_numpy(X) * 1e-4
    Y = torch.from_numpy(Y)

    if savepath is not None:
        os.makedirs(savepath, exist_ok=True)

        #gdf.to_file(os.path.join(savepath, "traintiles.shp"))

        with rio.open(os.path.join(savepath, "sentinel2.tif"), "w", **meta) as dst:
            dst.write(x_test.astype("uint16"))

        with rio.open(os.path.join(savepath, "existing_labels.tif"), "w", **labelprofile) as dst:
            dst.write(y_test.astype("uint16"), 1)

        gdf = gpd.read_file(os.path.join(savepath, "settlements.shp"))
        with rio.open(os.path.join(savepath, "sentinel2.tif"), "r") as src:
            buildings = rio.features.rasterize(gdf.to_crs(src.crs).geometry, out_shape=(src.width, src.height),
                                               transform=src.transform, all_touched=True)

        torch.save(dict(
            X=X,
            Y=Y,
            x_test=x_test,
            y_test=y_test,
            buildings=buildings,
            meta=meta),
            os.path.join(savepath, 'data.npz'))

    return X, Y, x_test, y_test, buildings, meta

def main():
    import matplotlib.pyplot as plt
    from skimage.exposure import equalize_hist

    # SEGMENTATION
    plt.tight_layout()
    plt.show()

    ds = SettlementDataset("/data/RepreSent/UC2", "37LBL", segmentation=True)
    X,Y = sample_batch(ds, target_index=200)

    fig, axs = plt.subplots(X.shape[0], 2, figsize=(3*2, 3*X.shape[0]))
    for x,y, axs_row in zip(X,Y, axs):
        ax = axs_row[0]
        ax.imshow(equalize_hist(x[np.array([3, 2, 1])]).transpose(1, 2, 0))
        ax.set_xticks([])
        ax.set_yticks([])

        ax = axs_row[1]
        ax.imshow(y)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()

    # CLASSIFICATION
    ds = SettlementDataset("/data/RepreSent/UC2", "37LBL", segmentation=False)
    X,Y = sample_batch(ds, target_index=200)

    fig, axs = plt.subplots(X.shape[0], 1, figsize=(3, 3*X.shape[0]))
    for x,y, ax in zip(X,Y, axs):
        ax.imshow(equalize_hist(x[np.array([3, 2, 1])]).transpose(1, 2, 0))
        ax.set_title("settlement" if y == 1 else "non-settlement")
        ax.set_xticks([])
        ax.set_yticks([])

if __name__ == '__main__':
    main()
