### ACCESS VIA EUMETSAT OSI SAF > OPeNDAP > THREDDS

from siphon.catalog import TDSCatalog
import requests
import re
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pyresample.geometry import SwathDefinition, GridDefinition
from pyresample.kd_tree import resample_nearest


class SST_HighRes:
    def __init__(self, min_lon=-26.5, max_lon=-24.5, min_lat=37, max_lat=38.5, start_dt="2020-01-01", end_dt="2020-01-02"):
        self.min_lon = min_lon
        self.max_lon = max_lon
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.start_dt = start_dt
        self.end_dt = end_dt

    def fetch_data(self, year="2020", day="001", before_2016=False):
        if before_2016:
            metOp = "OSI-204-metop_a"
        else:
            metOp = "OSI-204-b-metop_b"

        catalog_url = f"https://tds0.ifremer.fr/thredds/catalog/{metOp}/{year}/{day}/catalog.xml"
        cat = TDSCatalog(catalog_url)

        session = requests.Session()
        ds_list = []

        target_lat_range = (self.min_lat, self.max_lat)
        target_lon_range = (self.min_lon, self.max_lon)

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
                ds_candidate = xr.open_dataset(opendap_url, engine="pydap", session=session)
                ds_list.append(ds_candidate)

        return ds_list


    def regrid_to_common_grid(self, ds_list, resolution_km=1.0):
        lat_res_deg = resolution_km / 111.0
        center_lat = (self.min_lat + self.max_lat) / 2
        lon_res_deg = resolution_km / (111.0 * np.cos(np.radians(center_lat)))

        target_lons = np.arange(self.min_lon, self.max_lon, lon_res_deg)
        target_lats = np.arange(self.min_lat, self.max_lat, lat_res_deg)
        grid_lon, grid_lat = np.meshgrid(target_lons, target_lats)

        target_grid = GridDefinition(lons=grid_lon, lats=grid_lat)

        regridded_layers = []

        for dataset in ds_list:
            sst = dataset['sea_surface_temperature'].squeeze().values
            lat = dataset['lat'].values
            lon = dataset['lon'].values

            swath_def = SwathDefinition(lons=lon, lats=lat)

            regridded = resample_nearest(
                swath_def, sst, target_grid,
                radius_of_influence=5000,
                fill_value=np.nan
            )
            regridded_layers.append(regridded)

        stacked = np.stack(regridded_layers, axis=0)
        merged = np.nanmean(stacked, axis=0)

        return grid_lon, grid_lat, merged


    def get_feature_map(self, ds_list):
        grid_lon, grid_lat, merged = self.regrid_to_common_grid(ds_list)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        pc = ax.pcolormesh(grid_lon, grid_lat, merged, cmap='viridis', shading='auto')
        plt.colorbar(pc, ax=ax, label='SST (K)')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Merged High-Resolution SST composite - 1km')
        plt.tight_layout()
        plt.show()



# session = requests.Session()

# year = "2020"
# day = "001"

# before_2016 = False  # Set to True if you want to access data before 2016

# if before_2016:
#     metOp = "OSI-204-metop_a"
# else:
#     metOp = "OSI-204-b-metop_b"

# # Point siphon at the catalog.xml (not .html) for a given folder
# catalog_url = f"https://tds0.ifremer.fr/thredds/catalog/{metOp}/{year}/{day}/catalog.xml"
# cat = TDSCatalog(catalog_url)

# # List available datasets (files) in this catalog
# print(cat.datasets)

# target_lat_range = (37, 38.5)
# target_lon_range = (-26.5, -24.5)  # adjust if your actual target uses -180/180 convention

# ds = []

# def extract(attr_name, text):
#     match = re.search(rf'{attr_name}\s+([-\d.]+)', text)
#     return float(match.group(1)) if match else None

# # loop through datasets to find one that covers the target region
# for name, ds_ref in cat.datasets.items():

#     opendap_url = ds_ref.access_urls['OPENDAP']  # e.g. .../dodsC/OSI-204-b-metop_b/2020/001/xxxx.nc
#     das_url = opendap_url + ".das"

#     resp = requests.get(das_url, timeout=10)
#     resp.raise_for_status()  # raise an error if the request failed

#     text = resp.text

#     lat_min = extract('southernmost_latitude', text)
#     lat_max = extract('northernmost_latitude', text)
#     lon_min = extract('westernmost_longitude', text)
#     lon_max = extract('easternmost_longitude', text)

#     # print(f"{name}: lat {lat_min}-{lat_max}, lon {lon_min}-{lon_max}")

#     lat_ok = (float(lat_min) != -90) and (float(lat_min) <= target_lat_range[1]) and (float(lat_max) >= target_lat_range[0])
#     lon_ok = (float(lon_min) != -180) and (float(lon_min) <= target_lon_range[1]) and (float(lon_max) >= target_lon_range[0])

#     if lat_ok and lon_ok:
#         ds_candidate = xr.open_dataset(opendap_url, engine="pydap", session=session)
#         ds.append(ds_candidate)
#         print(f"{name}, {ds_candidate}")
#         print("MATCH")
#         print(f"{name}: lat {lat_min}-{lat_max}, lon {lon_min}-{lon_max}")

        
# print(f"Number of matching granules: {len(ds)}")

# ### OVERLAP LAYERS (DIFFERENT GRANULES) TO A COMMON GRID FOR PLOTTING
# ### REGRID TO 1KM RESOLUTION (AVHRR KNOWN RESOLUTION)

# resolution_km = 1.0  # known resolution of AVHRR

# # Latitude: straightforward, ~111 km per degree everywhere
# lat_res_deg = resolution_km / 111.0

# # Longitude: depends on latitude — use the center of your target box
# center_lat = (target_lat_range[0] + target_lat_range[1]) / 2
# lon_res_deg = resolution_km / (111.0 * np.cos(np.radians(center_lat)))

# target_lons = np.arange(target_lon_range[0], target_lon_range[1], lon_res_deg)
# target_lats = np.arange(target_lat_range[0], target_lat_range[1], lat_res_deg)
# grid_lon, grid_lat = np.meshgrid(target_lons, target_lats)

# target_grid = GridDefinition(lons=grid_lon, lats=grid_lat)

# regridded_layers = []

# for i, dataset in enumerate(ds):  # your list of matched/cropped granules
#     sst = dataset['sea_surface_temperature'].squeeze().values
#     lat = dataset['lat'].values
#     lon = dataset['lon'].values

#     swath_def = SwathDefinition(lons=lon, lats=lat)

#     regridded = resample_nearest(
#         swath_def, sst, target_grid,
#         radius_of_influence=5000,  # meters — how far to search for a source pixel per target cell
#         fill_value=np.nan
#     )
#     regridded_layers.append(regridded)

# ### MERGE REGRIDDING RESULTS (AVERAGE WHERE OVERLAP OCCURS)
# stacked = np.stack(regridded_layers, axis=0)
# merged = np.nanmean(stacked, axis=0)  # ignores NaNs automatically

# ### PLOTTING
# fig, ax = plt.subplots(figsize=(10, 6))
# pc = ax.pcolormesh(grid_lon, grid_lat, merged, cmap='viridis', shading='auto')
# plt.colorbar(pc, ax=ax, label='SST (K)')
# ax.set_xlabel('Longitude')
# ax.set_ylabel('Latitude')
# ax.set_title('Merged High-Resolution SST composite - 1km, ' + str(ds[0].time.values[0])[:10])
# plt.tight_layout()
# plt.show()


### IF YOU WANT TO PLOT EACH GRANULE INDIVIDUALLY, UNCOMMENT BELOW

# for i, dataset in enumerate(ds):
#     print(f"{dataset['lat'].values.min()} to {dataset['lat'].values.max()}, {dataset['lon'].values.min()} to {dataset['lon'].values.max()}")

#     lat = dataset['lat']
#     lon = dataset['lon']

#     mask = (
#         (lat >= target_lat_range[0]) & (lat <= target_lat_range[1]) &
#         (lon >= target_lon_range[0]) & (lon <= target_lon_range[1])
#     )

#     n_matching_pixels = int(mask.sum())
#     if n_matching_pixels == 0:
#         print(f"Dataset {i}: no overlap — skipping.")
#         continue

#     rows, cols = np.where(mask.values)
#     row_min, row_max = rows.min(), rows.max()
#     col_min, col_max = cols.min(), cols.max()
#     row_dim, col_dim = mask.dims

#     ds_subset = dataset.isel({
#         row_dim: slice(row_min, row_max + 1),
#         col_dim: slice(col_min, col_max + 1)
#     })

#     sst = ds_subset['sea_surface_temperature'].squeeze()  # drops the singleton time dim
#     # sst_celsius = sst - 273.15  # uncomment if your file is in Kelvin — check ds['sea_surface_temperature'].attrs for units

#     print(sst.dims, sst.shape)  # sanity check — should now be ('NUMROWS', 'NUMCELLS') or similar, (1080, 2048)

#     fig, ax = plt.subplots(figsize=(10, 6))
#     pc = ax.pcolormesh(ds_subset['lon'], ds_subset['lat'], sst, cmap='viridis', shading='auto')
#     plt.colorbar(pc, ax=ax, label=f"SST MetOp - 1km, dataset {i}, " + str(ds_subset.time.values[0])[:10])
#     ax.set_xlabel('Longitude')
#     ax.set_ylabel('Latitude')
#     plt.tight_layout()
#     plt.show()