import numpy as np
from pyresample.geometry import SwathDefinition, GridDefinition
from pyresample.kd_tree import resample_nearest
from datetime import date, datetime
import requests
import re
import xarray as xr
from siphon.catalog import TDSCatalog


def regrid_to_common_grid(ds_list, bbox, spec, resolution_km=1.0):
        """Regrid a list of swath datasets to a common grid defined by the bounding box and resolution."""
        dataset_name = spec.short_name
        lon0, lon1, lat0, lat1 = bbox

        # Calculate the target grid resolution in degrees based on the specified resolution in kilometers
        lat_res_deg = resolution_km / 111.0
        center_lat = (lat0 + lat1) / 2
        lon_res_deg = resolution_km / (111.0 * np.cos(np.radians(center_lat)))

        target_lons = np.arange(lon0, lon1, lon_res_deg)
        target_lats = np.arange(lat0, lat1, lat_res_deg)
        grid_lon, grid_lat = np.meshgrid(target_lons, target_lats)

        target_grid = GridDefinition(lons=grid_lon, lats=grid_lat)

        regridded_layers = []

        for dataset in ds_list:
            sst = dataset['sea_surface_temperature'].squeeze().values
            lat = dataset['latitude'].values
            lon = dataset['longitude'].values

            swath_def = SwathDefinition(lons=lon, lats=lat)

            regridded = resample_nearest(
                swath_def, sst, target_grid,
                radius_of_influence=5000,
                fill_value=np.nan
            )
            regridded_layers.append(regridded)

        stacked = np.stack(regridded_layers, axis=0)
        with np.errstate(all="ignore"):  # all-NaN slices are expected at the swath edges
            merged = np.nanmean(stacked, axis=0)

        return xr.Dataset(
            {dataset_name: (("latitude", "longitude"), merged)},
            coords={"latitude": target_lats, "longitude": target_lons},
        )


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


def get_data_url(bbox, catalog):

    """Return a list of OPeNDAP URLs from the IFREMER THREDDS catalog that intersect the bounding box."""
    lon0, lon1, lat0, lat1 = bbox
    target_lat_range = (lat0, lat1)
    target_lon_range = (lon0, lon1)

    url_list = []
    cat = TDSCatalog(catalog)

    def extract(attr_name, text):
        match = re.search(rf'{attr_name}\s+([-\d.]+)', text)
        return float(match.group(1)) if match else None

    for name, ds_ref in cat.datasets.items():
        opendap_url = ds_ref.access_urls['OPENDAP']
        das_url = opendap_url + ".das"

        resp = requests.get(das_url, timeout=10)
        resp.raise_for_status()

        text = resp.text

        lat_min = extract('southernmost_latitude', text)
        lat_max = extract('northernmost_latitude', text)
        lon_min = extract('westernmost_longitude', text)
        lon_max = extract('easternmost_longitude', text)

        lat_ok = (float(lat_min) != -90) and (float(lat_min) <= target_lat_range[1]) and (float(lat_max) >= target_lat_range[0])
        lon_ok = (float(lon_min) != -180) and (float(lon_min) <= target_lon_range[1]) and (float(lon_max) >= target_lon_range[0])

        if lat_ok and lon_ok:
            url_list.append(opendap_url)

    return url_list



    
