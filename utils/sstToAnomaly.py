# utils/climatology.py
import warnings
import numpy as np
import xarray as xr
from utils.caching import subset_path
from data.prepareGridData import to_target_grid

_GROUPS = {
    "day": "time.dayofyear",
    "month": "time.month",
    "season": "time.season",   # DJF, MAM, JJA, SON
    "year": "time.year",
}

def climatology(data, path, frequency="day"):
    """Mean of `data` (Dataset or DataArray) grouped by day/month/season/year."""
    if frequency not in _GROUPS:
        raise ValueError(f"freq must be one of {sorted(_GROUPS)}, got {frequency!r}")
    if data.sizes["time"] == 0:
        raise ValueError(f"No data inside the chosen period")

    clim = data.groupby(_GROUPS[frequency]).mean(dim="time")
    clim = to_target_grid(clim)
    clim.to_netcdf(path)
    return clim


# user_trange must at least be 365days/1 year for this to work
def get_sst_anomaly(data, spec, user_trange, full_trange, user_bbox, frequency="day"):
    """Anomaly of `data` relative to its own climatology, returned for the user's range/bbox.

    `data` must span several years (the baseline); `user_trange` can be any sub-range.
    """
    path = subset_path(spec=spec, bbox=user_bbox, trange=user_trange, wider_trange=full_trange, name="anomaly")
    if path.exists():
        return xr.load_dataset(path)

    data_variables = list(spec.variables)
    sst_data = data[data_variables[0]]

    clim_path = subset_path(spec=spec, bbox=user_bbox, trange=full_trange, name="clim")
    if clim_path.exists():
        clim = xr.load_dataset(clim_path)
    else:
        clim = climatology(sst_data, clim_path, frequency)       # baseline from ALL of `data`
    
    start_dt, end_dt = user_trange
    min_lon, max_lon, min_lat, max_lat = user_bbox
    subset = sst_data.sel(
        time = slice(start_dt, end_dt),
        longitude= slice(min_lon, max_lon),
        latitude= slice(min_lat, max_lat),
    )
    clim = clim.sel(
        longitude=slice(min_lon, max_lon), 
        latitude=slice(min_lat, max_lat)
    )

    sst_anomaly = subset.groupby(_GROUPS[frequency]) - clim
    sst_anomaly = to_target_grid(sst_anomaly)
    # sst_anomaly_ds = sst_anomaly.to_dataset(name="sst_anomaly")
    sst_anomaly.to_netcdf(path)

    return sst_anomaly

