import numpy as np
from pyresample.geometry import SwathDefinition, GridDefinition
from pyresample.kd_tree import resample_nearest
from datetime import date, datetime
from data.prepareGridData import make_target_grid
import requests
import re
import xarray as xr
from siphon.catalog import TDSCatalog
from concurrent.futures import ThreadPoolExecutor


# def regrid_to_common_grid(ds_list, bbox, spec, resolution_km=1.0):
#         """Regrid a list of swath datasets to a common grid defined by the bounding box and resolution."""
#         dataset_name = spec.short_name
#         lon0, lon1, lat0, lat1 = bbox

#         # Calculate the target grid resolution in degrees based on the specified resolution in kilometers
#         lat_res_deg = resolution_km / 111.0
#         center_lat = (lat0 + lat1) / 2
#         lon_res_deg = resolution_km / (111.0 * np.cos(np.radians(center_lat)))

#         target_lons = np.arange(lon0, lon1, lon_res_deg)
#         target_lats = np.arange(lat0, lat1, lat_res_deg)
#         grid_lon, grid_lat = np.meshgrid(target_lons, target_lats)

#         target_grid = GridDefinition(lons=grid_lon, lats=grid_lat)

#         regridded_layers = []

#         for dataset in ds_list:
#             sst = dataset['sea_surface_temperature'].squeeze().values
#             lat = dataset['latitude'].values
#             lon = dataset['longitude'].values

#             # --- diagnostics (remove once you've found the problem) ---
#             print("sst shape:", sst.shape, "| lon/lat shape:", lon.shape, lat.shape)
#             print("sst valid fraction:", np.isfinite(sst).mean())
#             print("swath lon:", np.nanmin(lon), np.nanmax(lon),
#                   "| lat:", np.nanmin(lat), np.nanmax(lat))
#             print("grid  lon:", target_lons.min(), target_lons.max(),
#                   "| lat:", target_lats.min(), target_lats.max())
#             # -----------------------------------------------------------

#             swath_def = SwathDefinition(lons=lon, lats=lat)

#             regridded = resample_nearest(
#                 swath_def, sst, target_grid,
#                 radius_of_influence=5000,
#                 fill_value=np.nan
#             )
#             print("resampled valid pixels:", np.isfinite(regridded).sum(), "\n")
#             regridded_layers.append(regridded)

#         stacked = np.stack(regridded_layers, axis=0)
#         with np.errstate(all="ignore"):  # all-NaN slices are expected at the swath edges
#             merged = np.nanmean(stacked, axis=0)

#         return xr.Dataset(
#             {dataset_name: (("latitude", "longitude"), merged)},
#             coords={"latitude": target_lats, "longitude": target_lons},
#         )


def regrid_to_common_grid(ds_list, bbox, spec, res=1/120):
    """Regrid swath datasets onto make_target_grid(bbox, res) and average them."""
    grid = make_target_grid(bbox, res)
    lats, lons = grid["latitude"].values, grid["longitude"].values
    target_grid = GridDefinition(*np.meshgrid(lons, lats))      # (lons, lats), both 2D

    layers = []
    for dataset in ds_list:
        sst = dataset["sea_surface_temperature"].squeeze().values
        swath_def = SwathDefinition(lons=dataset["longitude"].values,
                                    lats=dataset["latitude"].values)
        layers.append(resample_nearest(swath_def, sst, target_grid,
                                       radius_of_influence=5000, fill_value=np.nan))

    with np.errstate(all="ignore"):
        merged = np.nanmean(np.stack(layers, axis=0), axis=0)

    return xr.Dataset({spec.short_name: (("latitude", "longitude"), merged)},
                      coords={"latitude": lats, "longitude": lons})


METOP_A_END = date(2016, 1, 1)  # dates before this use metop_a


def get_catalog_url(date="2020-01-01T00:00:00"):

    """Build the IFREMER THREDDS catalog URL for a given date."""
    if isinstance(date, str):
        d = datetime.strptime(date, "%Y-%m-%d").date()
    elif isinstance(date, datetime):
        d = date.date()

    if d < METOP_A_END:
        metop = "OSI-204-metop_a"
    else:  # includes everything from 2026-01-01 on
        metop = "OSI-204-b-metop_b"

    print(f"get catalog: {metop}, {d:%Y}, {d:%j}")

    return (
        f"https://tds0.ifremer.fr/thredds/catalog/"
        f"{metop}/{d:%Y}/{d:%j}/catalog.xml"
    )


# def get_data_url(bbox, catalog):

#     """Return a list of OPeNDAP URLs from the IFREMER THREDDS catalog that intersect the bounding box."""
#     lon0, lon1, lat0, lat1 = bbox
#     target_lat_range = (lat0, lat1)
#     target_lon_range = (lon0, lon1)

#     url_list = []
#     cat = TDSCatalog(catalog)

#     def extract(attr_name, text):
#         match = re.search(rf'{attr_name}\s+([-\d.]+)', text)
#         return float(match.group(1)) if match else None

#     for name, ds_ref in cat.datasets.items():
#         opendap_url = ds_ref.access_urls['OPENDAP']
#         das_url = opendap_url + ".das"

#         resp = requests.get(das_url, timeout=10)
#         resp.raise_for_status()

#         text = resp.text

#         lat_min = extract('southernmost_latitude', text)
#         lat_max = extract('northernmost_latitude', text)
#         lon_min = extract('westernmost_longitude', text)
#         lon_max = extract('easternmost_longitude', text)

#         lat_ok = (float(lat_min) != -90) and (float(lat_min) <= target_lat_range[1]) and (float(lat_max) >= target_lat_range[0])
#         lon_ok = (float(lon_min) != -180) and (float(lon_min) <= target_lon_range[1]) and (float(lon_max) >= target_lon_range[0])

#         if lat_ok and lon_ok:
#             url_list.append(opendap_url)

#     return url_list

def _fetch_bounds(session, url):
    try:
        r = session.get(url + ".das", timeout=20)
        r.raise_for_status()
        vals = []
        for a in ("southernmost_latitude", "northernmost_latitude",
                  "westernmost_longitude", "easternmost_longitude"):
            m = re.search(rf'{a}\s+([-\d.]+)', r.text)
            vals.append(float(m.group(1)) if m else None)
        return url, vals
    except requests.RequestException:
        return url, None


def get_data_url(bbox, catalog, session, max_workers=16):
    lon0, lon1, lat0, lat1 = bbox
    cat = TDSCatalog(catalog)
    urls = [ds.access_urls["OPENDAP"] for ds in cat.datasets.values()]

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(lambda u: _fetch_bounds(session, u), urls))

    keep = []
    for url, b in results:
        if b is None or None in b:
            continue
        lat_min, lat_max, lon_min, lon_max = b
        lat_ok = lat_min != -90 and lat_min <= lat1 and lat_max >= lat0
        lon_ok = lon_min != -180 and lon_min <= lon1 and lon_max >= lon0
        if lat_ok and lon_ok:
            keep.append(url)
    return keep


    
