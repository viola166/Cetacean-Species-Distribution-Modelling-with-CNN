### ACCESS VIA AVISO > OpeNDAP

import requests
import xarray as xr

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


class FSLE:
    def __init__(self, min_lon=-26.5, max_lon=-24.5, min_lat=37, max_lat=38.5, start_dt="2001-01-01", end_dt="2001-12-31"):
        self.min_lon = min_lon
        self.max_lon = max_lon
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.start_dt = start_dt
        self.end_dt = end_dt

    def fetch_data(self):
        # Open the FSLE dataset from the THREDDS server using OPeNDAP
        # spatial resolution: 0.04 degree, temporal resolution: daily
        url = (
            "http://tds-odatis.aviso.altimetry.fr/"
            "thredds/dodsC/dataset-duacs-dt-global-allsat-madt-fsle"
        )

        session = requests.Session()
        ds = xr.open_dataset(
            url,
            engine="pydap",
            session=session,
        )
        return ds

    def get_subset(self):
        ds = self.fetch_data()
        subset = ds[["fsle_max", "theta_max", "lat_bnds", "lon_bnds", "crs"]].sel(
            time=slice(self.start_dt, self.end_dt),
            lat=slice(self.min_lat, self.max_lat),
            lon=slice(self.min_lon + 360, self.max_lon + 360),  # convert to 0-360 range
        )
        return subset

    def get_feature_map(self, fsle_slice):
        fig = plt.figure(figsize=(12, 6))
        ax = plt.axes(projection=ccrs.PlateCarree())

        im = fsle_slice.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap="viridis",          # or "Greys_r" — FSLE fronts often shown in grayscale
            cbar_kwargs={"label": "FSLE (1/day)"}
        )

        ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=2)
        ax.add_feature(cfeature.COASTLINE)
        ax.set_title(f"FSLE — 0.04°, {str(fsle_slice.time.values)[:10]}")
        plt.show()


# # Open the FSLE dataset from the THREDDS server using OPeNDAP
# # spatial resolution: 0.04 degree, temporal resolution: daily
# url = (
#     "http://tds-odatis.aviso.altimetry.fr/"
#     "thredds/dodsC/dataset-duacs-dt-global-allsat-madt-fsle"
# )

# session = requests.Session()

# ds = xr.open_dataset(
#     url,
#     engine="pydap",
#     session=session,
# )

# print(ds)

# # download a subset of the dataset for a specific time range and spatial extent
# subset = ds[["fsle_max", "theta_max", "lat_bnds", "lon_bnds", "crs"]].sel(
#     time=slice("2001-01-01", "2001-12-31"),
#     lat=slice(37, 38.5),
#     lon=slice(333.5, 335.5)
# )

# # Save the subset to a NetCDF file (format to store multi-dimensional scientific data)
# # subset.to_netcdf("fsle_subset.nc")


# # Pick a single time step to plot (e.g., 2001-01-01)
# fsle_slice = subset["fsle_max"].isel(time=0)

# fig = plt.figure(figsize=(12, 6))
# ax = plt.axes(projection=ccrs.PlateCarree())

# im = fsle_slice.plot(
#     ax=ax,
#     transform=ccrs.PlateCarree(),
#     cmap="viridis",          # or "Greys_r" — FSLE fronts often shown in grayscale
#     cbar_kwargs={"label": "FSLE (1/day)"}
# )

# ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=2)
# ax.add_feature(cfeature.COASTLINE)
# ax.set_title(f"FSLE — 0.04°, {str(fsle_slice.time.values)[:10]}")
# plt.show()

fsle = FSLE()
subset = fsle.get_subset()
print(subset)
# fsle.get_feature_map(subset)