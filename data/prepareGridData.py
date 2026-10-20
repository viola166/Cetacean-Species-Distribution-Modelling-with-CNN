import torch
import numpy as np
import xarray as xr
import pandas as pd
from scipy.stats import skew
from matplotlib import pyplot as plt


# Pick a target grid (e.g. the finest one, or a fixed grid you define)
step = 1 / 120  # 30 arc-seconds, ~0.00833 deg
n_lat = round(1.5 / step) + 1   # 181
n_lon = round(2.0 / step) + 1   # 241

TARGET_GRID = xr.Dataset(
    coords={
        "latitude": np.linspace(37.0, 38.5, n_lat),
        "longitude": np.linspace(-26.5, -24.5, n_lon),
    }
)

NUMBER_OF_PIXELS_PER_PATCH = 13  # Number of pixels per patch in each dimension (should be odd to have a center pixel)
SKEW_THRESHOLD = 1.0


def to_target_grid(ds, target=TARGET_GRID):
    return ds.interp_like(target, method="linear")   ### CONSIDER WHICH METHOD TO USE (LINEAR, NEAREST, ETC.) BASED ON DATA CHARACTERISTICS

def merge_datasets_common_grid(dataset_list):
    merged = xr.merge([to_target_grid(d, TARGET_GRID) for d in dataset_list])
    return merged

def ds_to_tensor(ds):
    # disregard depth if present: keep only the shallowest level
    if "depth" in ds.dims:
        ds = ds.sortby("depth").isel(depth=0, drop=True)

    x = ds.to_array("channel")  # (channel, time, lat, lon)
    x = x.transpose("time", "channel", "latitude", "longitude")
    return torch.from_numpy(x.values).float()  # (N, C, H, W)


def extract_patch(channel_ds, lat_coor, lon_coor, time, size=NUMBER_OF_PIXELS_PER_PATCH):
    """
    Extract a square patch of size (size x size) from the channel dataset at the specified lat/lon coordinates and time.
    Can handle one or more data variables.

    channel_ds : xarray Dataset with one or more data variables, dims
                 (time, latitude, longitude). Variables without a time dim
                 (e.g. static bathymetry) are also supported, as is a dataset
                 with no time dimension at all.
    lat_coor   : Latitude coordinate of the center of the patch
    lon_coor   : Longitude coordinate of the center of the patch
    time       : Time for which to extract the patch (nearest time step is used)
    size       : Size of the square patch (default is 13 > odd number ensures a center pixel)

    Returns
    -------
    patch      : xarray DataArray containing the extracted patch
    """

    lats = channel_ds["latitude"].values
    lons = channel_ds["longitude"].values
    n_lat, n_lon = len(lats), len(lons)

    # Nearest center pixel
    lat_idx = int(np.argmin(np.abs(lats - lat_coor)))
    lon_idx = int(np.argmin(np.abs(lons - lon_coor)))

    # Exactly `size` pixel indices per axis (may extend beyond the grid)
    lat_ids = np.arange(lat_idx - size // 2, lat_idx - size // 2 + size)
    lon_ids = np.arange(lon_idx - size // 2, lon_idx - size // 2 + size)

    lat_valid = (lat_ids >= 0) & (lat_ids < n_lat)
    lon_valid = (lon_ids >= 0) & (lon_ids < n_lon)

    # Select time (nearest step), if the dataset has a time dimension
    sel = {}
    if "time" in channel_ds.dims:
        t_idx = channel_ds.get_index("time").get_indexer(
            [pd.Timestamp(time)], method="nearest"      ### CONSIDER WHETHER TO USE 'nearest' OR 'pad/backfill' BASED ON DATA CHARACTERISTICS
        )[0]
        sel["time"] = int(t_idx)

    # Clip indices for isel, then blank out the out-of-grid pixels
    patch = channel_ds.isel(
        **sel,
        latitude=np.clip(lat_ids, 0, n_lat - 1),
        longitude=np.clip(lon_ids, 0, n_lon - 1),
    )

    # Restore proper coordinates (extrapolated outside the grid; assumes a regular grid)
    dlat = float(lats[1] - lats[0]) if n_lat > 1 else 1.0
    dlon = float(lons[1] - lons[0]) if n_lon > 1 else 1.0
    patch = patch.assign_coords(
        latitude=lats[0] + lat_ids * dlat,
        longitude=lons[0] + lon_ids * dlon,
    )

    inside = (
        xr.DataArray(lat_valid, dims="latitude", coords={"latitude": patch["latitude"]})
        & xr.DataArray(lon_valid, dims="longitude", coords={"longitude": patch["longitude"]})
    )
    patch = patch.where(inside)

    return patch



def get_patch_pixels(csv_data, channel_ds, size=NUMBER_OF_PIXELS_PER_PATCH):
    """Collect all finite pixel values inside the training patches, per variable.
    Returns {variable_name: 1D float64 array}."""
    times, lats, lons = _valid_records(channel_ds=channel_ds, csv_data=csv_data)
    samples = {v: [] for v in channel_ds.data_vars}

    for t, lat, lon in zip(times, lats, lons):
        patch = extract_patch(channel_ds, lat, lon, t, size=size)
        for v in samples:
            vals = patch[v].values.astype(np.float64).ravel()
            samples[v].append(vals[np.isfinite(vals)])  # drops land/missing/out-of-grid

    out = {}
    for v, chunks in samples.items():
        arr = np.concatenate(chunks) if chunks else np.array([])
        if arr.size == 0:
            raise ValueError(f"All patch pixels for '{v}' are NaN.")
        out[v] = arr
    return out


def _valid_records(channel_ds, csv_data):
    
    """Parse records and keep those whose center lies inside the dataset's extent.
    Returns (times, lats, lons) as aligned pandas Series."""
    df = pd.DataFrame.from_dict(csv_data, orient="index")

    times = pd.to_datetime(
        df["eventDate"].astype(str).str.split("/").str[0],
        errors="coerce",
        utc=True,
    ).dt.tz_localize(None)
    lats = pd.to_numeric(df["decimalLatitude"], errors="coerce")
    lons = pd.to_numeric(df["decimalLongitude"], errors="coerce")

    valid = (
        times.notna() & lats.notna() & lons.notna()
        & (lats >= float(channel_ds["latitude"].min()))
        & (lats <= float(channel_ds["latitude"].max()))
        & (lons >= float(channel_ds["longitude"].min()))
        & (lons <= float(channel_ds["longitude"].max()))
    )

    if "time" in channel_ds.dims:
        t_vals = channel_ds["time"].values
        t_tol = np.median(np.diff(t_vals)) / 2 if len(t_vals) > 1 else np.timedelta64(12, "h")
        valid &= (
            (times >= pd.Timestamp(t_vals.min() - t_tol))
            & (times <= pd.Timestamp(t_vals.max() + t_tol))
        )

    if not valid.any():
        raise ValueError("No training records fall inside the dataset's time/space extent.")

    return times[valid], lats[valid], lons[valid]


def get_normalization_stats(csv_data, channel_ds, size=NUMBER_OF_PIXELS_PER_PATCH):
    """
    Get per-variable normalization statistics, computed only over the pixels of
    the patches around the training records (not the whole feature map).

    csv_dict   : Dictionary containing the csv training dataset
                 {key: {"originalEventID", "scientificName", "eventDate",
                        "decimalLatitude", "decimalLongitude"}}
    channel_ds : xarray Dataset with one or more data variables, dims
                 (time, latitude, longitude). Variables without a time dim
                 (e.g. static bathymetry) are supported.
    size       : Patch edge length in pixels. Must match the size used for
                 the model inputs (default 12).

    Returns
    -------
    stats : dict {variable_name: (mean, std)}, both floats.
    """
    var_names = list(channel_ds.data_vars)
    if not var_names:
        raise ValueError("channel_ds has no data variables.")

    # only keep csv records that lie within the dataset's extent
    times_valid, lats_valid, lons_valid = _valid_records(csv_data=csv_data, channel_ds=channel_ds)

    # --- accumulate sums over all valid patch pixels, per variable ---
    n = {v: 0 for v in var_names}
    s = {v: 0.0 for v in var_names}
    ss = {v: 0.0 for v in var_names}

    ### PROBABLY EXCHANGE EXTRACT_PATCH FOR GET_PATCH_PIXELS
    for t, lat, lon in zip(times_valid, lats_valid, lons_valid):
        patch = extract_patch(channel_ds, lat, lon, t, size=size)
        for v in var_names:
            vals = patch[v].values.astype(np.float64).ravel()
            vals = vals[np.isfinite(vals)]  # drops land, missing, and out-of-grid pixels
            n[v] += vals.size
            s[v] += vals.sum()
            ss[v] += np.square(vals).sum()

    # --- final mean / std per variable ---
    stats = {}
    for v in var_names:
        if n[v] == 0:
            raise ValueError(f"All patch pixels for '{v}' are NaN.")
        mean = s[v] / n[v]
        std = float(np.sqrt(max(ss[v] / n[v] - mean ** 2, 0.0)))
        if std < 1e-8:  # constant channel: avoid division by zero later
            std = 1.0
        stats[v] = (float(mean), std)

    return stats


def normalize_channel_variables(channel_ds, stats):
    """
    Normalize every data variable of channel_ds with its own training statistics.

    channel_ds : xarray Dataset with one or more data variables
    stats      : dict {variable_name: (mean, std)} from get_normalization_stats

    Returns
    -------
    norm_ds : Dataset, z-scored variables with NaNs filled with 0 (= training mean), float32
    mask_ds : Dataset, 1 = valid pixel, 0 = land/missing, float32
                (variables are named "<var>_mask")
    """
    missing = [v for v in channel_ds.data_vars if v not in stats]
    if missing:
        raise KeyError(f"No normalization stats for variables: {missing}")

    norm_vars, mask_vars = {}, {}
    for v in channel_ds.data_vars:
        da = channel_ds[v]
        mean, std = stats[v]
        mask_vars[f"{v}_mask"] = da.notnull().astype("float32")
        norm_vars[v] = ((da - mean) / std).fillna(0.0).astype("float32")

    return xr.Dataset(norm_vars), xr.Dataset(mask_vars)


def normalize_channel(channel_ds, csv_data, split, size=NUMBER_OF_PIXELS_PER_PATCH, verbose=True):
    """
    Fit normalization stats on the training records of one fold and apply them
    to the channel's grid.

    channel_ds : xarray Dataset (one or more data variables) on the common grid
    csv_data   : dict {key: record} with all occurrence records
    split      : one entry of data_splitter.split(csv_data),
                 with keys "fold", "train", "val", "test" (lists of record keys)
    size       : patch size in pixels, must match extract_patch

    Returns
    -------
    norm_ds : Dataset, z-scored variables, NaNs filled with 0, float32
    mask_ds : Dataset, 1 = valid pixel, 0 = land/missing/out-of-grid, float32
    stats   : dict {variable_name: (mean, std)}, fitted on the training records only
    """
    train = {k: csv_data[k] for k in split["train"]}

    # get the normalization statistics only based on the training set 
    stats = get_normalization_stats(channel_ds=channel_ds, csv_data=train, size=size)
    if verbose:
        for var_name, (mean, std) in stats.items():
            print(f"Fold {split['fold']}, {var_name}: mean={mean:.4f}, std={std:.4f}")

    norm_ds, mask_ds = normalize_channel_variables(channel_ds, stats)
    return norm_ds, mask_ds, stats


# def get_skewness(channel_ds, csv_data, fold, size=NUMBER_OF_PIXELS_PER_PATCH):
#     """Skewness per variable over the training patch pixels of one fold."""
#     train = {k: csv_data[k] for k in fold["train"]}
#     pixels = get_patch_pixels(train, channel_ds, size=size)
#     return {v: float(skew(x)) for v, x in pixels.items()}

# # handle skewness
# def log_transform(channel_ds, eps=1e-3):
#     """log10 transform for strictly positive, right-skewed variables (e.g. chlorophyll)."""
#     out = channel_ds.copy()
#     for v in out.data_vars:
#         out[v] = np.log10(out[v].clip(min=eps))   # clip guards against zeros/negatives
#     return out


def _apply_transform(x, kind, eps=1e-3):
    """Pointwise transform; works on numpy arrays and xarray DataArrays."""
    if kind == "none":
        return x
    if kind == "log10":        # strictly positive, right-skewed (e.g. chlorophyll)
        return np.log10(x.clip(min=eps))
    if kind == "log1p":        # non-negative with zeros
        return np.log1p(x.clip(min=0))
    if kind == "signed_log":   # signed, heavy-tailed (e.g. bathymetry)
        return np.sign(x) * np.log1p(np.abs(x))
    raise ValueError(f"Unknown transform: {kind}")


def remove_skewness(channel_ds, csv_data, split, size=NUMBER_OF_PIXELS_PER_PATCH,
                    threshold=SKEW_THRESHOLD, eps=1e-3, transforms=None, verbose=True):
    """
    Evaluate the skewness of each variable over the training patch pixels and
    apply a transform where needed.

    transforms : optional dict {variable: kind}. If given, no decision is made
                 and these transforms are applied as they are (use this to
                 reuse a decision across folds).

    Returns
    -------
    out_ds : Dataset with the transformed variables
    chosen : dict {variable: kind}, kind in {"none", "log10", "log1p", "signed_log"}
    """
    train = {k: csv_data[k] for k in split["train"]}
    pixels = get_patch_pixels(train, channel_ds, size=size)

    out_ds = channel_ds.copy()
    chosen = {}

    for v, x in pixels.items():
        if transforms is not None:
            kind = transforms[v]
        else:
            s_before = float(skew(x))
            kind = "none"

            if s_before > threshold:
                # pick the candidate by the range of the training pixels
                if x.min() > 0:
                    kind = "log10"
                elif x.min() == 0:
                    kind = "log1p"
                else:
                    kind = "signed_log"

                # keep it only if it reduces |skew|; the transform is pointwise,
                # so applying it to the pixel array equals applying it to the grid
                s_after = float(skew(_apply_transform(x, kind, eps)))
                if abs(s_after) >= abs(s_before):
                    kind = "none"
                if verbose:
                    print(f"{v}: skew {s_before:.2f} -> {s_after:.2f} with {kind if kind != 'none' else 'log (rejected)'}")
            elif verbose:
                msg = f"{v}: skew {s_before:.2f}, no transform"
                if s_before < -threshold:
                    msg += " (left-skewed, not handled)"
                print(msg)

        out_ds[v] = _apply_transform(channel_ds[v], kind, eps)
        chosen[v] = kind

    return out_ds, chosen


def histogram(channel_ds, csv_data, fold):
    train = {k: csv_data[k] for k in fold["train"]}
    pixels = get_patch_pixels(train, channel_ds)

    vars = channel_ds.data_vars
    for var in vars:
        x = pixels[var]
        print(np.percentile(x, [0.1, 1, 5, 50, 95, 99, 99.9]), x.max())
        print("share of pixels > 3 sigma:", np.mean((x - x.mean()) / x.std() > 3))
        plt.hist(x, bins=80); plt.show()


def make_patches(norm_dss, mask_dss, lat, lon, time, size=NUMBER_OF_PIXELS_PER_PATCH):
    """
    Loops over all channel datasets (and their mask datasets, if given) and extracts the
    patch of every variable around one sample.

    Fills the NaNs that extract_patch introduces (land, missing data, off-grid) with 0.
    In the normalized datasets 0 equals the training mean, so this is mean imputation.
    Converts xarrays to plain numpy arrays (dropping coordinates and labels).

    norm_dss : list of normalized channel Datasets
    mask_dss : list of mask Datasets (same order and length as norm_dss), or None.
               If None, no mask channels are added (mean imputation only).
    lat, lon, time : centre and time of the sample
    size     : patch edge length in pixels

    Returns
    -------
    np.ndarray float32 of shape
        (C, size, size)   if mask_dss is None
        (2C, size, size)  otherwise, interleaved per dataset:
                          dataset 1 values, dataset 1 masks, dataset 2 values, ...
    with C the total number of variables across all datasets.
    """
    if mask_dss is not None and len(mask_dss) != len(norm_dss):
        raise ValueError("mask_dss must have the same length as norm_dss (or be None).")

    chans = []
    for k, nds in enumerate(norm_dss):
        p = extract_patch(nds, lat, lon, time, size).fillna(0.0)   # off-grid -> 0 (= training mean)
        chans += [p[v].values for v in nds.data_vars]

        if mask_dss is not None:
            mds = mask_dss[k]
            m = extract_patch(mds, lat, lon, time, size).fillna(0.0)   # off-grid -> 0 (= invalid)
            chans += [m[v].values for v in mds.data_vars]

    return np.stack(chans).astype(np.float32)                     # (2C, size, size)

