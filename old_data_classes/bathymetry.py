### ACCESS VIA GEBCO > OPeNDAP

import requests
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

class Bathymetry:
    def __init__(self, min_lon=-26.5, max_lon=-24.5, min_lat=37, max_lat=38.5):
        self.min_lon = min_lon
        self.max_lon = max_lon
        self.min_lat = min_lat
        self.max_lat = max_lat

    def fetch_data(self):
        # Open the bathymetry dataset from the THREDDS server using OPeNDAP
        # spatial resolution: 15-sec arc, temporal resolution: static
        url = (
            "https://dap.ceda.ac.uk/"
            "thredds/dodsC/bodc/gebco/global/gebco_2026/sub_ice_topography_bathymetry/netcdf/GEBCO_2026_sub_ice.nc"
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
        subset = ds[["lon", "lat", "crs", "elevation"]].sel(
            lon=slice(self.min_lon, self.max_lon),
            lat=slice(self.min_lat, self.max_lat),
        )
        return subset

    def get_feature_map(self, subset):
        fig = plt.figure(figsize=(12, 6))
        ax = plt.axes(projection=ccrs.PlateCarree())

        # diverging colormap centered on sea level (0 m) so land/ocean read correctly
        norm = mcolors.TwoSlopeNorm(
            vmin=float(subset["elevation"].min()), vcenter=0, vmax=float(subset["elevation"].max())
        )

        im = subset["elevation"].plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap="terrain",              # or cmocean.cm.topo if you have cmocean installed
            norm=norm,
            cbar_kwargs={"label": "Elevation (m)"},
        )

        ax.add_feature(cfeature.COASTLINE)
        ax.set_title("Bathymetry - 15-sec arc, 2026")
        plt.show()

# # Open the bathymetry dataset from the THREDDS server using OPeNDAP
# # spatial resolution: 15-sec arc, temporal resolution: static
# url = (
#     "https://dap.ceda.ac.uk/"
#     "thredds/dodsC/bodc/gebco/global/gebco_2026/sub_ice_topography_bathymetry/netcdf/GEBCO_2026_sub_ice.nc"
# )

# session = requests.Session()

# ds = xr.open_dataset(
#     url,
#     engine="pydap",
#     session=session,
# )

# print(ds)

# # download a subset of the dataset for the specified spatial extent
# subset = ds[["lon", "lat", "crs", "elevation"]].sel(
#     lon=slice(-26.5, -24.5),
#     lat=slice(37, 38.5),
# )


# fig = plt.figure(figsize=(12, 6))
# ax = plt.axes(projection=ccrs.PlateCarree())

# # diverging colormap centered on sea level (0 m) so land/ocean read correctly
# norm = mcolors.TwoSlopeNorm(
#     vmin=float(subset["elevation"].min()), vcenter=0, vmax=float(subset["elevation"].max())
# )

# im = subset["elevation"].plot(
#     ax=ax,
#     transform=ccrs.PlateCarree(),
#     cmap="terrain",              # or cmocean.cm.topo if you have cmocean installed
#     norm=norm,
#     cbar_kwargs={"label": "Elevation (m)"},
# )

# ax.add_feature(cfeature.COASTLINE)
# ax.set_title("Bathymetry - 15-sec arc, 2026")
# plt.show()