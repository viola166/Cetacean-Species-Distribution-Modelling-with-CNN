# utils/climatology.py
import warnings
import numpy as np
import xarray as xr
from utils.caching import subset_path
from utils.dataCropper import nearest_date
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
def get_sst_anomaly(data, spec, user_trange, full_trange, user_bbox, frequency="day", refresh=False, train_spec=None):
    """Anomaly of `data` relative to its own climatology, returned for the user's range/bbox.

    `data` must span several years (the baseline); `user_trange` can be any sub-range.
    """
    path = subset_path(spec=spec, bbox=user_bbox, trange=user_trange, wider_trange=full_trange, name="anomaly")
    if path.exists() and refresh is False:
        return xr.load_dataset(path)

    data_variable = list(spec.variables)[0]
    sst_data = data[[data_variable]] 

    if train_spec is None:
        clim_path = subset_path(spec=spec, bbox=user_bbox, trange=full_trange, name="clim")
    else:
        clim_path = subset_path(spec=train_spec, bbox=user_bbox, trange=full_trange, name="clim")       # ensures that for clim_period data from comprehensive dataset is loaded
    
    if clim_path.exists() and refresh is False:
        clim = xr.load_dataset(clim_path)
    else:
        clim = climatology(sst_data, clim_path, frequency)       # baseline from ALL of `data`
    
    subset = nearest_date(sst_data, user_trange)

    min_lon, max_lon, min_lat, max_lat = user_bbox
    subset = subset.sel(longitude=slice(min_lon, max_lon),
                        latitude=slice(min_lat, max_lat))

    clim = clim.sel(
        longitude=slice(min_lon, max_lon), 
        latitude=slice(min_lat, max_lat)
    )

    sst_anomaly = subset.groupby(_GROUPS[frequency]) - clim
    sst_anomaly = to_target_grid(sst_anomaly)
    # sst_anomaly_ds = sst_anomaly.to_dataset(name="sst_anomaly")
    sst_anomaly.to_netcdf(path)

    return sst_anomaly

