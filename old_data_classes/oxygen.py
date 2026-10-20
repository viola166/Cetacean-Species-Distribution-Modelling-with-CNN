### ACCESS VIA Copernicus > Copernicus Marine Toolbox
import copernicusmarine as cm
import matplotlib.pyplot as plt

class Oxygen:
    def __init__(
        self,
        dataset_id="cmems_mod_glo_bgc_my_0.25deg_P1D-m",
        variables=["o2"],
        minimum_longitude=-26.5,
        maximum_longitude=-24.5,
        minimum_latitude=37,
        maximum_latitude=38.5,
        start_datetime="2020-01-01T00:00:00",
        end_datetime="2021-01-01T00:00:00"
    ):
        self.dataset_id = dataset_id
        self.variables = variables
        self.min_lon = minimum_longitude
        self.max_lon = maximum_longitude
        self.min_lat = minimum_latitude
        self.max_lat = maximum_latitude
        self.start_dt = start_datetime
        self.end_dt = end_datetime

    def fetch_data(self):
        ds = cm.open_dataset(
            dataset_id=self.dataset_id,
            variables=self.variables,
            minimum_longitude=self.min_lon,
            maximum_longitude=self.max_lon,
            minimum_latitude=self.min_lat,
            maximum_latitude=self.max_lat,
            start_datetime=self.start_dt,
            end_datetime=self.end_dt,
        )
        return ds

    def get_field(self, time_index=0, depth_index=0):
        ds = self.fetch_data()
        field = ds.o2.isel(time=time_index, depth=depth_index)  # surface only (depth index 0 = shallowest level)
        return field

    def get_feature_map(self, field):
        fig, ax = plt.subplots(figsize=(10, 6))
        pc = ax.pcolormesh(field['longitude'], field['latitude'], field, cmap='viridis', shading='auto')
        plt.colorbar(pc, ax=ax, label='O2 Concentration (mol/m³)')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title("Mole Concentration of Dissolved Molecular Oxygen in Sea Water - 0.25°, " + str(field.time.values[0])[:10])
        plt.show()