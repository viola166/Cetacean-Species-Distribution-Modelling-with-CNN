import numpy as np
import xarray as xr
from pathlib import Path
import csv
import pandas as pd
import warnings
from data.prepareGridData import get_normalization_stats, TARGET_GRID
import json


NUM_PIXELS_PER_TILE = 20  # Number of pixels per tile in each dimension
NUM_PIXELS_BUFFER = 1 # Number of pixels to buffer (as padding) around each tile to reduce information leakage between tiles
NUM_FOLDS = 7 # Number of folds for cross-validation
SPLIT_DATA_PATH = Path("output", "Split_CSV_Data", "spatial_block_cv.json")

step = 1 / 120  # 30 arc-seconds, ~0.00833 deg
n_lat = round(1.5 / step) + 1   # 181
n_lon = round(2.0 / step) + 1   # 241

TARGET_GRID = xr.Dataset(
    coords={
        "latitude": np.linspace(37.0, 38.5, n_lat),
        "longitude": np.linspace(-26.5, -24.5, n_lon),
    }
)

EVENT_TABLE_NAME = "event.txt"
OCCURRENCE_TABLE_NAME = "occurrence.txt"


class DataSplitter:
    """uses spatial_block_cv() to split the data into training, validation, and test sets based on spatial blocks (tiles)."""

    def __init__(self, n_folds=NUM_FOLDS, n_pixels=NUM_PIXELS_PER_TILE, spatial_buffer=NUM_PIXELS_BUFFER, use_val=True, seed=42):
        self.n_folds = n_folds
        self.n_pixels = n_pixels
        self.spatial_buffer = spatial_buffer
        self.use_val = use_val
        self.seed = seed

    # def split(self, data):
    #     """
    #     Split the data into training, validation, and test sets based on spatial blocks (tiles).
    #     Returns a list of dictionaries, each containing the keys 'fold', 'train', 'val', and 'test'.
    #     """

    #     tile_id = self.assign_tiles_to_pixels(n_pixels=self.n_pixels, spatial_buffer=self.spatial_buffer)
    #     return self.spatial_block_cv(data, tile_id, self.n_folds, self.use_val, self.seed)


    # grid is subdivided into tiles of TILE_PIXELS_PER_TILE x TILE_PIXELS_PER_TILE degrees, and each data point 
    # is assigned to a tile based on its lat/lon coordinates.
    def assign_tiles_to_pixels(self, target_grid=TARGET_GRID, n_pixels=NUM_PIXELS_PER_TILE, spatial_buffer=NUM_PIXELS_BUFFER):
        """
        Assign every (lat, lon) grid cell (from TARGET_GRID) to a square tile of n_pixels x n_pixels.

        n_pixels       : tile side length in pixels (int, or (lat_px, lon_px))
        spatial_buffer : margin in pixels (float/int, or (lat_px, lon_px)). Cells within
                        this distance of a tile edge that borders another tile are dropped.
                        0.0 = no buffer.

        Returns
        -------
        tile_id : (lat, lon) DataArray of tile ids (float), NaN for dropped cells.
                Use tile_id.notnull() as the "kept" mask.
        """
        n_lat, n_lon = (n_pixels, n_pixels) if np.isscalar(n_pixels) else n_pixels
        b_lat, b_lon = (spatial_buffer, spatial_buffer) if np.isscalar(spatial_buffer) else spatial_buffer

        # if the buffer is too large, no tile remains, so raise an error
        if 2 * b_lat >= n_lat or 2 * b_lon >= n_lon:
            raise ValueError("spatial_buffer must be smaller than half of n_pixels")

        def axis_info(size, n, b):
            """
            assigns tile indices along one axis and returns a boolean mask
            of which points are kept (not dropped due to buffer)
            """
            pos = np.arange(size)
            tile_idx = pos // n                       # tile index along this axis
            k = pos % n                          # position inside the tile (0 .. n-1)
            # Only trim edges that touch a neighbouring tile, not the outer domain border
            has_lower_neighbour = tile_idx > 0                                      # tile (tile_idx) has neighbor with lower index (not first tile)
            has_upper_neighbour = (tile_idx * n + n) < size                         # tile (tile_idx) has neighbor with higher index (not last tile)
            too_close_lower = has_lower_neighbour & (k < b)                         # too close to lower edge/"left end" of tile (and has a lower neighbour); meaning the pixel is within the buffer distance of the tile edge that borders another tile > gets dropped
            too_close_upper = has_upper_neighbour & ((n - 1 - k) < b)               # too close to upper edge/"right end" of tile (and has an upper neighbour)
            return tile_idx, ~(too_close_lower | too_close_upper)


        lat_idx, lat_ok = axis_info(target_grid.sizes["latitude"], n_lat, b_lat)
        lon_idx, lon_ok = axis_info(target_grid.sizes["longitude"], n_lon, b_lon)

        # from 1D tile indices along each axis, compute a unique 2D tile id for each (lat, lon) cell
        n_tiles_lon = lon_idx.max() + 1                                             # number of tile columns
        ids = lat_idx[:, None] * n_tiles_lon + lon_idx[None, :]                     # unique tile id for each (lat, lon) cell (2D), based on its tile row and column
        keep = lat_ok[:, None] & lon_ok[None, :]

        return xr.DataArray(
            np.where(keep, ids, np.nan),
            coords={"latitude": target_grid["latitude"], "longitude": target_grid["longitude"]},
            dims=("latitude", "longitude"),
            name="tile_id",
        )


    def assign_tiles_to_targets(self, target_dic, tile_id):
        """
        Assign each target to a tile based on its lat/lon coordinates and the tile_id DataArray.

        Parameters
        ----------
        target_dic : dict of targets, as returned by load_target_data()
        tile_id     : (lat, lon) DataArray of tile ids (float), NaN for dropped cells.

        Returns
        -------
        dict {key: {"originalEventID", "scientificName", "eventDate",
                    "decimalLatitude", "decimalLongitude", "tile_id"}}
        """
        # Convert target_dic to a DataFrame for easier processing
        df = pd.DataFrame.from_dict(target_dic, orient="index")

        # Create a function to find the nearest grid point in the tile_id DataArray
        def get_tile_id(lat, lon):
            try:
                return float(tile_id.sel(latitude=lat, longitude=lon, method="nearest").values)
            except KeyError:
                return float("nan")  # Return NaN if the coordinates are out of bounds

        # Apply the function to each row in the DataFrame
        df["tile_id"] = df.apply(lambda row: get_tile_id(row["decimalLatitude"], row["decimalLongitude"]), axis=1)

        # Convert back to dictionary format
        return df.to_dict("index")


    def assign_tiles_to_datapts(self, data, tile_id):
        """
        Look up the tile of every datapoint. Returns {key: tile_id (int)}.
        Datapoints outside the grid, or inside a dropped buffer cell, are left out.

        data :      dict {key: {"originalEventID", "scientificName", "eventDate",
                    "decimalLatitude", "decimalLongitude"}}
        tile_id :   (lat, lon) DataArray of tile ids (float), NaN for dropped cells.
        
        Returns
        -------
        dict {key: tile_id (int)} for datapoints that are inside the grid and not in a dropped buffer cell.
        """
        keys = list(data)
        lats = np.array([data[k]["decimalLatitude"] for k in keys], dtype=float)
        lons = np.array([data[k]["decimalLongitude"] for k in keys], dtype=float)

        # nearest grid cell for every datapoint
        pixel = tile_id.sel(
            {"latitude": xr.DataArray(lats, dims="p"), "longitude": xr.DataArray(lons, dims="p")},
            method="nearest",
        )
        # a point further than half a cell from its nearest cell centre is outside the grid
        half_lat = np.abs(np.median(np.diff(tile_id["latitude"].values))) / 2 + 1e-6
        half_lon = np.abs(np.median(np.diff(tile_id["longitude"].values))) / 2 + 1e-6
        inside = (np.abs(pixel["latitude"].values - lats) <= half_lat) & \
                (np.abs(pixel["longitude"].values - lons) <= half_lon)
        tiles = pixel.values
        in_buffer = inside & np.isnan(tiles)

        if (~inside).any() or in_buffer.any():
            warnings.warn(f"{int((~inside).sum())} datapoints lie outside the grid and "
                        f"{int(in_buffer.sum())} lie in buffer cells; all were dropped.")
        return {k: int(t) for k, t, ok in zip(keys, tiles, inside & ~np.isnan(tiles)) if ok}


    def assign_folds_to_tiles(self, tile_of_key, n_folds, seed=42):
        """
        Whole tiles -> folds, balancing the number of datapoints per fold.

        tile_of_key : dict {key: tile_id (int)}
                      Tile of each datapoint, as returned by assign_tiles_to_datapts().
                      Only tiles that contain at least one datapoint take part.
        n_folds     : int
                      Number of folds. Must not exceed the number of non-empty tiles,
                      otherwise some fold would be empty.
        seed        : int
                      Seed for the random tie-breaking between tiles of equal size.

        Returns
        -------
        fold_of_tile : dict {tile_id (int): fold (int, 0 .. n_folds - 1)}
        
        """
        tiles, counts = np.unique(list(tile_of_key.values()), return_counts=True)
        if n_folds > len(tiles):
            raise ValueError(f"n_folds={n_folds} but only {len(tiles)} tiles contain datapoints.")
        rng = np.random.default_rng(seed)
        order = sorted(rng.permutation(len(tiles)), key=lambda i: -counts[i])  # big tiles first
        load = np.zeros(n_folds)
        fold_of_tile = {}
        for i in order:
            f = int(np.argmin(load))            # fold with the fewest datapoints so far
            fold_of_tile[int(tiles[i])] = f
            load[f] += counts[i]
        return fold_of_tile


    def spatial_block_cv(self, csv_data, n_folds=7, use_val=True, seed=42, path=SPLIT_DATA_PATH, refresh=False):
        """
        Spatial block cross-validation. Returns a list with one dict per fold:
        {"fold", "train", "val", "test"}, each a list of keys of `data`.
        Fold f uses fold group (group of tiles) f as test and fold (f+1) % n_folds as validation (if use_val).
        """
        if path.exists() and not refresh:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)

        if use_val and n_folds < 3:
            raise ValueError("use_val=True needs n_folds >= 3.")
        tile_id = self.assign_tiles_to_pixels()                                 # get the tile_id of each pixel from the TARGET_GRID
        tile_of_dptKey = self.assign_tiles_to_datapts(csv_data, tile_id)        # get the tile_id of each record key (datapoint)
        fold_of_tile = self.assign_folds_to_tiles(tile_of_dptKey, n_folds, seed) # get the tile group (fold) of each fold
        fold_of_key = {dpt_key: fold_of_tile[tile] for dpt_key, tile in tile_of_dptKey.items()} # get the tile group (fold) of each record key (datapoint)

        splits = []
        for f in range(n_folds):
            val_f = (f + 1) % n_folds if use_val else None
            s = {"fold": f, "train": [], "val": [], "test": []}
            for k, fk in fold_of_key.items():
                name = "test" if fk == f else "val" if fk == val_f else "train"
                s[name].append(k)
            splits.append(s)

        # save (write to a temp file first, so a crash can't leave a half-written cache)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(splits, f)
        tmp.replace(path)

        return splits
