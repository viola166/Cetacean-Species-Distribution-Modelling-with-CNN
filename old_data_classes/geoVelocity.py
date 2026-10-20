### ACCESS VIA Copernicus > Copernicus Marine Toolbox

import copernicusmarine as cm
import matplotlib.pyplot as plt
import hashlib
from pathlib import Path

import copernicusmarine as cm
import matplotlib.pyplot as plt
import xarray as xr


### ACCESS VIA Copernicus > Copernicus Marine Toolbox
class GeoVelocity:
    def __init__(
        self,
        dataset_id="cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D",
        variables=["ugos", "vgos", "ugosa", "vgosa"],
        min_lon=-26.5,
        max_lon=-24.5,
        min_lat=37,
        max_lat=38.5,
        start_dt="2020-01-01T00:00:00",
        end_dt="2021-01-01T00:00:00",
        cache_dir="geoVelo_subsets",
    ):
        self.dataset_id = dataset_id
        self.variables = variables
        self.min_lon = min_lon
        self.max_lon = max_lon
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.start_dt = start_dt
        self.end_dt = end_dt
        self.cache_dir = Path("output", cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ cache
    @property
    def _day(self):
        """Unique per dataset/variables/region/period, so changing any of them gives a fresh cache."""
        h = hashlib.md5((self.dataset_id + "|".join(self.variables)).encode()).hexdigest()[:6]
        return (
            f"{self.start_dt[:10]}_{self.end_dt[:10]}"
            f"_lon{self.min_lon}_{self.max_lon}_lat{self.min_lat}_{self.max_lat}_{h}"
        )

    def _path(self, name):
        return self.cache_dir / f"{name}_{self._day}.nc"

    # ------------------------------------------------------------------- data
    def fetch_data(self, refresh=False):
        """Raw data: downloaded once from Copernicus, then read from the local cache."""
        path = self._path("raw")
        if path.exists() and not refresh:
            return xr.load_dataset(path)

        ds = cm.open_dataset(
            dataset_id=self.dataset_id,
            variables=self.variables,
            minimum_longitude=self.min_lon,
            maximum_longitude=self.max_lon,
            minimum_latitude=self.min_lat,
            maximum_latitude=self.max_lat,
            start_datetime=self.start_dt,
            end_datetime=self.end_dt,
        ).load()  # actually download now, not lazily
        ds.to_netcdf(path)
        return ds


    # def get_climatology(self, refresh=False):
    #     """Daily climatological mean (by day of year) of ALL variables."""
    #     path = self._path("clim")
    #     if path.exists() and not refresh:
    #         return xr.load_dataset(path)

    #     ds = self.fetch_data()
    #     clim = ds.groupby("time.dayofyear").mean(dim="time")
    #     clim.to_netcdf(path)
    #     return clim

    # def fetch_eke_data(self, refresh=False):
    #     """Eddy kinetic energy from anomalies relative to the daily climatology."""
    #     path = self._path("eke")
    #     if path.exists() and not refresh:
    #         return xr.load_dataarray(path)

    #     ds = self.fetch_data()
    #     clim = self.get_climatology()
    #     u_anom = ds.ugos.groupby("time.dayofyear") - clim.ugos
    #     v_anom = ds.vgos.groupby("time.dayofyear") - clim.vgos
    #     eke = (0.5 * (u_anom**2 + v_anom**2)).rename("eke")
    #     eke.to_netcdf(path)
    #     return eke

    # ------------------------------------------------------------------ plots
    def get_fields(self):
        ds = self.fetch_data()
        abs_geoVelo_eastWest_field = (ds.ugos.isel(time=0), "Absolut East-West Geostrophic Velocity")
        abs_geoVelo_northSouth_field = (ds.vgos.isel(time=0), "Absolut North-South Geostrophic Velocity")
        geoVelo_anomaly_eastWest_field = (ds.ugosa.isel(time=0), "Anomaly East-West Geostrophic Velocity")
        geoVelo_anomaly_northSouth_field = (ds.vgosa.isel(time=0), "Anomaly North-South Geostrophic Velocity")

        return (
            abs_geoVelo_eastWest_field,
            abs_geoVelo_northSouth_field,
            geoVelo_anomaly_eastWest_field,
            geoVelo_anomaly_northSouth_field,
        )


    def get_feature_map(self, fields, label="Velocity (m/s)", cmap="viridis"):
        # fields must be a list/tuple of tuples (field, title), also when it's only one field
        for field, title in fields:
            print(field.dims, field.shape)
            # use the field's own date if it has one, otherwise fall back to start_dt
            date = str(field["time"].values)[:10] if "time" in field.coords else str(self.start_dt)[:10]
            fig, ax = plt.subplots(figsize=(10, 6))
            pc = ax.pcolormesh(field["longitude"], field["latitude"], field, cmap=cmap, shading="auto")
            plt.colorbar(pc, ax=ax, label=label)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_title(title + " - 0.125°, " + date)
            plt.show()

 
    # def get_eke_map(self, date):
    #     """Map of EKE for a single day, e.g. get_eke_map("2020-01-01")."""
    #     eke = self.get_eke()
    #     field = eke.sel(time=date, method="nearest")
    #     self.get_feature_map([(field, "Eddy Kinetic Energy")], label="EKE (m²/s²)", cmap="magma")


if __name__ == "__main__":
    geo_vel = GeoVelocity(start_dt="1993-01-01T00:00:00", end_dt="2026-01-16T00:00:00")
    clim = geo_vel.get_climatology()  # first call downloads + computes, later calls just read from ./output
    eke = geo_vel.get_eke()


# ds = cm.open_dataset(
#     dataset_id = "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D",
#     variables = ["ugos", "vgos", "ugosa", "vgosa"],
#     minimum_longitude=-26.5,
#     maximum_longitude=-24.5,
#     minimum_latitude=37,
#     maximum_latitude=38.5,
#     start_datetime="2020-01-01T00:00:00",
#     end_datetime="2021-01-01T00:00:00",
# )

# print(ds.attrs)

# abs_geoVelo_eastWest_field = (ds.ugos.isel(time=0), "Absolut East-West Geostrophic Velocity")
# abs_geoVelo_northSouth_field = (ds.vgos.isel(time=0), "Absolut North-South Geostrophic Velocity")
# geoVelo_anomaly_eastWest_field = (ds.ugosa.isel(time=0), "Anomaly East-West Geostrophic Velocity")
# geoVelo_anomaly_northSouth_field = (ds.vgosa.isel(time=0), "Anomaly North-South Geostrophic Velocity")

# fields = [abs_geoVelo_eastWest_field, abs_geoVelo_northSouth_field, geoVelo_anomaly_eastWest_field, geoVelo_anomaly_northSouth_field]


# for field, title in fields:
#     print(field.dims, field.shape)  # should now be ('latitude', 'longitude') (7, 9)
#     fig, ax = plt.subplots(figsize=(10, 6))
#     pc = ax.pcolormesh(field['longitude'], field['latitude'], field, cmap='viridis', shading='auto')
#     plt.colorbar(pc, ax=ax, label='Velocity (m/s)')
#     ax.set_xlabel('Longitude')
#     ax.set_ylabel('Latitude')
#     ax.set_title(title + " - 0.125°, " + str(ds.time.values[0])[:10])
#     plt.show()