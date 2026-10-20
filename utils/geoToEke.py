import xarray as xr
from utils.caching import subset_path

_GROUPS = {
    "day": "time.dayofyear",
    "month": "time.month",
    "season": "time.season",   # DJF, MAM, JJA, SON
    "year": "time.year",
}

def climatology(data, path, frequency="day"):
    """Mean of `data` grouped by day/month/season/year (pure function, no I/O)."""
    if frequency not in _GROUPS:
        raise ValueError(f"frequency must be one of {sorted(_GROUPS)}, got {frequency!r}")
    if data.sizes["time"] == 0:
        raise ValueError("No data to compute a climatology from")
    
    clim = data.groupby(_GROUPS[frequency]).mean(dim="time")
    clim.to_netcdf(path)
    return clim

def anomaly(data, clim, frequency="day"):
    """`data` minus a precomputed climatology of the same frequency."""
    if frequency == "year":
        raise ValueError("Anomalies need a repeating cycle: use 'day', 'month' or 'season'.")
    out = data.groupby(_GROUPS[frequency]) - clim
    # drop the leftover group coordinate (dayofyear / month / season)
    return out.drop_vars(_GROUPS[frequency].split(".")[1], errors="ignore")


def get_eke(data, spec, user_trange, full_trange, user_bbox, frequency="day"):
    """Eddy kinetic energy from velocity anomalies relative to the climatology.

    `data`: Dataset covering the full baseline period (fetched by load_any).
    The climatology uses ALL of `data`'s time steps; only the returned
    anomalies are restricted to `user_trange` and `user_bbox`.
    """
    path = subset_path(spec=spec, bbox=user_bbox, trange=user_trange, wider_trange=full_trange, name="eke")
    if path.exists():
        return xr.load_dataset(path)

    data_variables = list(spec.variables)
    
    u_name, v_name = data_variables
    min_lon, max_lon, min_lat, max_lat = user_bbox
    start_dt, end_dt = user_trange

    # 1. crop space first: EKE is pointwise, so no margin is needed (should be redundant but is inexpensive)
    geo_data = data[[u_name, v_name]].sel(longitude=slice(min_lon, max_lon),
                                     latitude=slice(min_lat, max_lat))

    # 2. baseline from the FULL time range of the (small) box
    clim_path = subset_path(spec=spec, bbox=user_bbox, trange=full_trange, name="clim")
    if clim_path.exists():
        clim = xr.load_dataset(clim_path)
    else:
        clim = climatology(geo_data, clim_path, frequency)

    # 3. crop time, then subtract the baseline
    subset = geo_data.sel(time=slice(start_dt, end_dt))
    if subset.sizes["time"] == 0:
        raise ValueError(f"No data inside {user_trange}")
    anom = anomaly(subset, clim, frequency)

    # 4. EKE, returned as a Dataset to keep the Dataset -> Dataset contract
    eke = (0.5 * (anom[u_name]**2 + anom[v_name]**2)).rename("eke")
    eke_ds = eke.to_dataset()
    eke_ds.to_netcdf(path)
    return eke_ds