### ACCESS VIA Copernicus > Copernicus Marine Toolbox

import copernicusmarine as cm
import matplotlib.pyplot as plt

class pH:
    def __init__(
        self,
        dataset_id="cmems_mod_glo_bgc_my_0.25deg_P1M-m",
        variables=["ph"],
        minimum_longitude=-26.5,
        maximum_longitude=-24.5,
        minimum_latitude=37,
        maximum_latitude=38.5, 
        start_dt="2020-01-01T00:00:00", 
        end_dt="2021-01-01T00:00:00"
        ):
        self.dataset_id = dataset_id
        self.variables = variables
        self.min_lon = minimum_longitude
        self.max_lon = maximum_longitude
        self.min_lat = minimum_latitude
        self.max_lat = maximum_latitude
        self.start_dt = start_dt
        self.end_dt = end_dt
        self._ds = None


    def fetch_data(self):
        if self._ds is None:
            self._ds = cm.open_dataset(
                dataset_id=self.dataset_id,
                variables=self.variables,
                minimum_longitude=self.min_lon,
                maximum_longitude=self.max_lon,
                minimum_latitude=self.min_lat,
                maximum_latitude=self.max_lat,
                start_datetime=self.start_dt,
                end_datetime=self.end_dt,
            )
        return self._ds
    
    def get_array(self, depth_index=0): # surface only
        """Full (time, lat, lon) DataArray for one variable."""
        dataA = self.fetch_data()[self.variables[0]]
        if "depth" in dataA.dims:
            dataA = dataA.isel(depth=depth_index)
        return dataA

    def get_field(self, time_index=0, depth_index=0):   # keep for plotting
        return self.get_array(depth_index).isel(time=time_index)

    def get_feature_map(self, field):
        fig, ax = plt.subplots(figsize=(10, 6))
        pc = ax.pcolormesh(field['longitude'], field['latitude'], field, cmap='viridis', shading='auto')
        plt.colorbar(pc, ax=ax, label='pH')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title("Sea Water pH Reported on Total Scale - 0.25°, " + str(field.time.values)[:10])
        plt.show()


pH = pH()
pH_field = pH.get_field()
pH.get_feature_map(pH_field)