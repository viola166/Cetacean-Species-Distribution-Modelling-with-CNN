import numpy as np

def standardize_coords(ds):
    mapping = {"lat": "latitude", "lon": "longitude"}
    return ds.rename({k: v for k, v in mapping.items() if k in ds.dims or k in ds.coords})


def standardize_longitude(ds):
    """Wrap longitudes to -180..180 and sort ascending."""
    if "longitude" not in ds.coords:
        return ds
    lon = ds["longitude"]
    if float(lon.min()) >= -180 and float(lon.max()) <= 180:
        return ds                                   # already in range, nothing to do
    wrapped = ((lon + 180) % 360) - 180
    return ds.assign_coords({"longitude": wrapped}).sortby("longitude")