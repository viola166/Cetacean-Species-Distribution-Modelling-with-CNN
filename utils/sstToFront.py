import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
import xarray as xr
from utils.cayulaCornillon import CCA_SIED
from utils.caching import subset_path
from utils.dateConverter import date_to_index
from data.prepareGridData import to_target_grid
import pandas as pd


def fronts_to_mask(xdata_final, ydata_final, field):
    """
    Rasterize front point coordinates onto field's grid as a boolean mask.

    Parameters
    ----------
    xdata_final, ydata_final : 1D arrays
        Front point coordinates (lon, lat) returned by CCA_SIED.
    field : xarray.DataArray
        The SST field the fronts were detected on, dims ('lat', 'lon').

    Returns
    -------
    np.ndarray (bool), shape = field.shape
        True at grid cells nearest to a detected front point.
    """
    lat_vals = field['latitude'].values         ### SLIGHT CHANGE: 'lat' to 'latitude'
    lon_vals = field['longitude'].values        ### SLIGHT CHANGE: 'lon' to 'longitude'

    mask = np.zeros(field.shape, dtype=bool)

    for x, y in zip(xdata_final, ydata_final):
        i_lat = np.abs(lat_vals - y).argmin()
        i_lon = np.abs(lon_vals - x).argmin()
        mask[i_lat, i_lon] = True

    return mask

def _pixel_km(field):
    """Pixel size in km (dy, dx) at the field's mean latitude."""
    lat = field['latitude'].values
    lon = field['longitude'].values
    dy = np.abs(np.mean(np.diff(lat))) * 111.0
    dx = np.abs(np.mean(np.diff(lon))) * 111.0 * np.cos(np.deg2rad(lat.mean()))
    return dy, dx


def front_distance_channel(xdata_final, ydata_final, field):
    # """
    # Dense distance-to-nearest-front raster, same shape as field.

    # Parameters
    # ----------
    # normalize : {'log', 'minmax', None}
    #     'log'    -> log1p(distance), then min-max to [0, 1]  (recommended;
    #                 compresses the long right tail of far-from-front cells)
    #     'minmax' -> raw distance min-max scaled to [0, 1]
    #     None     -> raw distance in grid-cell units

    # Returns
    # -------
    # np.ndarray (float64), shape = field.shape
    #     No NaNs. 0 (or near-0) at front pixels, increasing with distance.
    #     If no fronts were detected in this image, returns an array of NaNs
    #     of the same shape (there is no meaningful distance field to compute).
    # """
    # mask = fronts_to_mask(xdata_final, ydata_final, field)

    # if not mask.any():
    #     return np.zeros(field.shape, dtype='float64')

    # # distance_transform_edt gives distance to the nearest True pixel from
    # # every False pixel (0 at True pixels themselves). Pass sampling so
    # # unequal lat/lon pixel spacing is respected in the distance calc.
    # lat_vals = field['latitude'].values
    # lon_vals = field['longitude'].values
    # dlat = np.abs(np.mean(np.diff(lat_vals)))
    # dlon = np.abs(np.mean(np.diff(lon_vals)))

    # dist = distance_transform_edt(~mask, sampling=(dlat, dlon))

    # if normalize == 'log':
    #     dist = np.log1p(dist)
    #     dist = (dist - dist.min()) / (np.ptp(dist) + 1e-12)
    # elif normalize == 'minmax':
    #     dist = (dist - dist.min()) / (np.ptp(dist) + 1e-12)
    # # else: leave as raw distance

    # return dist

    """
    Distance to the nearest front, in km.
    No fronts detected -> the patch's maximum possible distance everywhere
    ("as far from a front as possible"). Pair with front_present_channel().
    Not rescaled: z-scoring happens downstream on the whole training set.
    """
    mask = fronts_to_mask(xdata_final, ydata_final, field)
    dy, dx = _pixel_km(field)
    ny, nx = field.shape[-2:]
    if mask.any():
        return distance_transform_edt(~mask, sampling=(dy, dx))
    return np.full((ny, nx), np.hypot(ny * dy, nx * dx), dtype='float64')


def front_density_channel(xdata_final, ydata_final, field, sigma=2.0):
    """
    Gaussian-smoothed front density. Fixed scale: an isolated front pixel peaks
    at 1.0, overlapping neighbours can exceed it, and no-front images stay at 0.
    """
    mask = fronts_to_mask(xdata_final, ydata_final, field)
    return gaussian_filter(mask.astype('float64'), sigma=sigma) * (2 * np.pi * sigma**2)


def sst_gradient_channel(field):
    """SST gradient magnitude in K/km (NaNs preserved). No per-image scaling."""
    sst = field.values.astype('float64')
    dy, dx = _pixel_km(field)
    grad_y, grad_x = np.gradient(sst, dy, dx)
    return np.hypot(grad_x, grad_y)

def build_front_covariate_stack(xdata_final, ydata_final, field,
                                 include_density=True, include_gradient=True,
                                 density_sigma=2.0):
    """
    Convenience wrapper: build a (C, H, W) stack of front-derived CNN
    covariate channels, ready to concatenate with your other environmental
    covariates.

    Returns
    -------
    dict with keys among {'distance', 'density', 'gradient'}, each an
    np.ndarray of shape field.shape. Stack with e.g.:
        np.stack([channels['distance'], channels['gradient']], axis=0)
    """
    channels = {
        'distance': front_distance_channel(
            xdata_final, ydata_final, field
        )
    }

    if include_density:
        channels['density'] = front_density_channel(
            xdata_final, ydata_final, field, sigma=density_sigma
        )

    if include_gradient:
        channels['gradient'] = sst_gradient_channel(field)

    return channels


def get_front_channels(data, spec, user_bbox, wider_bbox, user_trange):
    # final-product cache: stored in the same form it is returned (Dataset)
    path = subset_path(spec=spec, bbox=user_bbox, trange=user_trange, name="frontCh")
    if path.exists():
        return xr.load_dataset(path)

    data_variables = list(spec.variables)

    # time-crop first (cheap, saves running CCA on unneeded steps),
    # but keep the wider bbox: CCA needs the margin
    w_min_lon, w_max_lon, w_min_lat, w_max_lat = wider_bbox
    data = data.sel(
        time=slice(*user_trange),
        latitude=slice(w_min_lat, w_max_lat),      # redundant if data is already the wider box
        longitude=slice(w_min_lon, w_max_lon),
    )
    if data.sizes["time"] == 0:
        raise ValueError(f"No data inside {user_trange}")

    per_step = []
    # loop over the time dimension > create one 2D datagrid per time index
    for i in range(data.sizes["time"]):
        sst_field_big = data[data_variables[0]].isel(time=i)
        day = pd.Timestamp(sst_field_big["time"].values).strftime("%Y-%m-%d")

        # per-day cache of the sparse front points, keyed on the wider box CCA ran on
        cca_path = subset_path(spec=spec, bbox=wider_bbox, trange=(day, day), name="cca")
        cca_path = cca_path.with_suffix(".npz")

        if cca_path.exists():
            with np.load(cca_path) as f:
                front_xdata, front_ydata = f["x"], f["y"]
        else:
            # sparse (x, y) front points; keep a length-1 time dim for CCA_SIED
            front_xdata, front_ydata = CCA_SIED(data.isel(time=[i]))
            np.savez(cca_path, x=np.asarray(front_xdata), y=np.asarray(front_ydata))

        # dict of three (H, W) rasters: distance, gradient, density
        front_channels = build_front_covariate_stack(front_xdata, front_ydata, sst_field_big)

        stack = np.stack(list(front_channels.values()), axis=0)       # (3, H, W)
        per_step.append(
            xr.DataArray(
                stack,
                dims=("channel", *sst_field_big.dims),
                coords={"channel": list(front_channels.keys()), **sst_field_big.coords},
                name="front_channels",
            )
        )

    # scalar "time" coord of each frame becomes the concat dimension
    da = xr.concat(per_step, dim="time").transpose("time", "channel", ...)

    # crop to the user's box only now, after detection used the margin
    min_lon, max_lon, min_lat, max_lat = user_bbox
    da = da.sel(latitude=slice(min_lat, max_lat), longitude=slice(min_lon, max_lon))

    # one variable per channel, matching the "outputs" contract used elsewhere
    out = da.to_dataset(dim="channel")
    out = to_target_grid(out)
    out.to_netcdf(path)
    return out


def get_front_frequency(data, spec, user_bbox, wider_bbox, user_trange, full_trange, window=30, smooth_sigma=1.0):
    # final-product cache: stored in the same form it is returned (Dataset)
    path = subset_path(spec=spec, bbox=user_bbox, trange=user_trange, name="frontFreq")
    if path.exists():
        return xr.load_dataset(path)

    t0, t1 = (pd.Timestamp(t) for t in full_trange)

    # crop to the wider (padded) box for detection, and pad the start in time
    # so the first requested day still has a full window
    w_min_lon, w_max_lon, w_min_lat, w_max_lat = wider_bbox
    ds_pad = data.sel(
        time=slice(t0, t1),
        latitude=slice(w_min_lat, w_max_lat),                   # should be redundant
        longitude=slice(w_min_lon, w_max_lon),
    )
    if ds_pad.sizes["time"] == 0:
        raise ValueError(f"No data inside {full_trange}")
    sst = ds_pad.analysed_sst

    masks, valids = [], []
    for i in range(sst.sizes["time"]):
        sst_i = sst.isel(time=i)
        day = pd.Timestamp(sst_i["time"].values).strftime("%Y-%m-%d")

        # cache key = the wider box CCA actually ran on
        cca_path = subset_path(spec=spec, bbox=wider_bbox, trange=(day, day), name="cca")
        cca_path = cca_path.with_suffix(".npz")

        if cca_path.exists():
            with np.load(cca_path) as f:
                front_xdata, front_ydata = f["x"], f["y"]
        else:
            front_xdata, front_ydata = CCA_SIED(ds_pad.isel(time=[i]))
            np.savez(cca_path, x=np.asarray(front_xdata), y=np.asarray(front_ydata))

        m = fronts_to_mask(front_xdata, front_ydata, sst_i).astype(float)
        if smooth_sigma:
            m = gaussian_filter(m, sigma=smooth_sigma)
        masks.append(m)
        valids.append(np.isfinite(sst_i.values).astype(float))

    masks  = xr.DataArray(np.stack(masks),  dims=sst.dims, coords=sst.coords)
    valids = xr.DataArray(np.stack(valids), dims=sst.dims, coords=sst.coords)

    # rolling window over time; min_periods=1 clips at the dataset start
    counts = masks.rolling(time=window, min_periods=1).sum()
    valid  = valids.rolling(time=window, min_periods=1).sum()
    freq = (counts / valid.where(valid > 0)).rename("front_frequency")

    # drop the time padding, then crop the spatial margin down to the user's box
    min_lon, max_lon, min_lat, max_lat = user_bbox
    freq = freq.sel(
        time=slice(t0, t1),
        latitude=slice(min_lat, max_lat),
        longitude=slice(min_lon, max_lon),
    )
    freq_ds = freq.to_dataset(name="front_frequency")
    freq_ds = to_target_grid(freq_ds)
    freq_ds.to_netcdf(path)
    return freq_ds
