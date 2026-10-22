import numpy as np
import xarray as xr
from scipy import ndimage
from scipy.spatial import cKDTree
from data.prepareGridData import to_target_grid
from utils.caching import subset_path

R_EARTH_KM = 6371.0088

def _to_xyz(lon, lat):
    lon, lat = np.deg2rad(lon), np.deg2rad(lat)
    return np.column_stack([
        R_EARTH_KM * np.cos(lat) * np.cos(lon),
        R_EARTH_KM * np.cos(lat) * np.sin(lon),
        R_EARTH_KM * np.sin(lat),
    ])


def get_dist_to_coast(ds, user_trange, user_bbox, spec):
    """Distance (km) from every grid cell to the nearest land/ocean transition.

    Static field: user_trange is ignored.
    """
    path = subset_path(spec=spec, bbox=user_bbox, trange=user_trange, name="dist")
    if path.exists():
        return xr.load_dataset(path)


    opts = spec.options
    elev = ds[spec.variables[0]].transpose("latitude", "longitude")
    lat, lon = elev.latitude.values, elev.longitude.values
    land = elev.values > opts.get("land_threshold", 0.0)

    # coast cells = land/ocean transitions, taken from both sides
    land_edge = land & ~ndimage.binary_erosion(land, border_value=1)
    ocean_edge = ~land & ~ndimage.binary_erosion(~land, border_value=1)
    iy, ix = np.nonzero(land_edge | ocean_edge)

    if iy.size == 0:
        print(f"WARNING: no coastline in domain for {spec.short_name}, returning NaN")
        dist = np.full(land.shape, np.nan)
    else:
        tree = cKDTree(_to_xyz(lon[ix], lat[iy]))
        LON, LAT = np.meshgrid(lon, lat)
        chord, _ = tree.query(_to_xyz(LON.ravel(), LAT.ravel()), workers=-1)
        dist = 2 * R_EARTH_KM * np.arcsin(np.clip(chord / (2 * R_EARTH_KM), 0, 1))
        dist = dist.reshape(land.shape)
        if opts.get("signed", False):
            dist = np.where(land, -dist, dist)

    out = xr.Dataset(
        {spec.short_name: (("latitude", "longitude"), dist.astype("float32"))},
        coords={"latitude": lat, "longitude": lon},
        attrs={"units": "km", "source": "derived from GEBCO land/ocean mask"},
    )

    # crop back to the requested bbox (relevant if wide_bbox was used for padding)
    lon0, lon1, lat0, lat1 = user_bbox
    out = out.sel(longitude=slice(lon0, lon1), latitude=slice(lat0, lat1))
    out = to_target_grid(out)
    out.to_netcdf(path)
    return out 

