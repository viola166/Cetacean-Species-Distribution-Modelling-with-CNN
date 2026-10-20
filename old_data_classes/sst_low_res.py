### ACCESS VIA Copernicus > Copernicus Marine Toolbox

import copernicusmarine as cm
import matplotlib.pyplot as plt
import xarray as xr
import hashlib
from pathlib import Path

class SST_LowRes:
    def __init__(
        self,
        dataset_id="METOFFICE-GLO-SST-L4-REP-OBS-SST",
        variables=["analysed_sst"],
        minimum_longitude=-26.5,
        maximum_longitude=-24.5,
        minimum_latitude=37,
        maximum_latitude=38.5, 
        start_dt="2020-01-01T00:00:00", 
        end_dt="2021-01-01T00:00:00",
        cache_dir="sstLowRes_subsets",
        ):
        self.dataset_id = dataset_id
        self.variables = variables
        self.min_lon = minimum_longitude
        self.max_lon = maximum_longitude
        self.min_lat = minimum_latitude
        self.max_lat = maximum_latitude
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
        ).load()

        ds.to_netcdf(path)
        return ds

    def get_field(self, time_index=0):
        ds = self.fetch_data()
        field_tuple = (ds.analysed_sst.isel(time=time_index), "Foundation Sea Surface Temperature")  # surface SST, first timestep
        return field_tuple

    def get_feature_map(self, field_tuple):
        field, title = field_tuple
        plt.figure(figsize=(10, 6))
        field.plot(cmap="viridis")
        # use the field's own date if it has one, otherwise fall back to start_dt
        date = str(field["time"].values)[:10] if "time" in field.coords else str(self.start_dt)[:10]
        plt.title(f"{title} - 0.05°, " + date)
        plt.show()

    def get_climatology(self, refresh=False):
        """Daily climatological mean (by day of year) of ALL variables."""
        path = self._path("clim")
        if path.exists() and not refresh:
            return xr.load_dataset(path)

        ds = SST_LowRes.fetch_data(self)
        clim = ds.groupby("time.dayofyear").mean(dim="time")
        clim.to_netcdf(path)
        return clim


# ds = cm.open_dataset(
#     dataset_id = "METOFFICE-GLO-SST-L4-REP-OBS-SST",
#     variables = ["analysed_sst"],
#     minimum_longitude=-26.5,
#     maximum_longitude=-24.5,
#     minimum_latitude=37,
#     maximum_latitude=38.5,
#     start_datetime="2020-01-01T00:00:00",
#     end_datetime="2021-01-01T00:00:00",
# )

# # print(ds.data_vars)

# import matplotlib.pyplot as plt

# field = ds.analysed_sst.isel(time=0)  # surface SST, first timestep

# plt.figure(figsize=(10, 6))
# field.plot(cmap="viridis")
# plt.title("Foundation Sea Surface Temperature - 0.05°, " + str(ds.time.values[0])[:10])
# plt.show()