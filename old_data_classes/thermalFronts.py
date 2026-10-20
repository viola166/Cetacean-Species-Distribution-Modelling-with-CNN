from utils.cayulaCornillon import CCA_SIED
from utils.sstToFront import fronts_to_mask
from utils.dateConverter import date_to_index
from data.sst_low_res import SST_LowRes
import matplotlib.pylab as plt
from utils.sstToFront import build_front_covariate_stack, fronts_to_mask
from scipy.ndimage import gaussian_filter
import numpy as np
import xarray as xr


class ThermalFronts(SST_LowRes):
    def __init__(self, *args, **kwargs):
        # initialize Thermal Fronts object with larger area, so CCA works > then crop
        kwargs["minimum_longitude"] = -27.5
        kwargs["maximum_longitude"] = -23.5
        kwargs["minimum_latitude"] = 35.8
        kwargs["maximum_latitude"] = 39.8
        super().__init__(*args, **kwargs)
        

    def fetch_data(self):
        return super().fetch_data()

    # field = fitered data 
    def get_field(self, time_index=0):

        # get the complete SST data
        ds_big = self.fetch_data()
        sst_field_big = ds_big.analysed_sst.isel(time=time_index)

        # use Cayula Cornillon Algorithm to retrieve all (x,y) coordinates where fronts are present (sparse point cloud)
        # returns two flat arrays
        front_xdata, front_ydata = CCA_SIED(ds_big.isel(time=[time_index]))

        # converting scattered front-point coordinates into three full, dense, (H, W)-shaped rasters aligned to field's grid
        # three rasters: distance from front, gradient, density
        # returns a dict with those three keys, each mapping to a 2D array
        front_channels = build_front_covariate_stack(front_xdata, front_ydata, sst_field_big)

        # Converts the dict of three separate (H, W) arrays into one (3, H, W) array
        front_stack = np.stack(list(front_channels.values()), axis=0) 

        # transform into a data array (from numpy array)
        front_channels_da = xr.DataArray(
            front_stack,
            dims=("channel", *sst_field_big.dims),                                      # has channel dimension + whatever dimensions 'field' has
            coords={"channel": list(front_channels.keys()), **sst_field_big.coords},    # values along each axis: distance, density, gradient + lat, lon, time
            name="front_channels",
        )

        # distance_field = front_channels_da.sel(channel="distance", latitude=slice(37.0, 38.5), longitude=slice(-26.0, -25.0))   # or however you're extracting it
        # n_nan = np.isnan(distance_field.values).sum()
        # n_total = distance_field.values.size
        # print(n_nan, "/", n_total)

        return front_channels_da.sel(latitude=slice(37.0, 38.5), longitude=slice(-26.5, -24.5))

    # get frequency of thermal field over the last 30 days (rolling window)
    def get_frequency_field(self, date="2020-10-01T00:00:00", window=30, smooth_sigma=1.0):
        ds_big = self.fetch_data()
        sst_field_today = ds_big.analysed_sst.sel(time=date)

        time_index = date_to_index(ds_big, date)

        counts = np.zeros(sst_field_today.shape)
        valid = np.zeros(sst_field_today.shape)

        first_day = time_index - window + 1
        if first_day <= 0:
            first_day = 0
        last_day = time_index + 1

        for day in range(first_day, last_day):
            sst_field_i = ds_big.analysed_sst.isel(time=day)
            ds_field_i = ds_big.isel(time=[day])
            front_xdata, front_ydata = CCA_SIED(ds_field_i)
            plt.close('all')

            mask = fronts_to_mask(front_xdata, front_ydata, sst_field_i)

            m = mask.astype(float)
            if smooth_sigma:
                m = gaussian_filter(m, sigma=smooth_sigma)
            counts += m
            valid += np.isfinite(sst_field_i.values)

        frequency = np.where(valid > 0, counts / np.maximum(valid, 1), np.nan)
        front_freq_field = sst_field_today.copy(data=frequency)
        front_freq_field.name = "front_frequency"
        return front_freq_field.sel(latitude=slice(37.0, 38.5), longitude=slice(-26.5, -24.5))


    def get_feature_map_old(self, field, channel="distance"):
        plt.figure(figsize=(10, 6))
        field.sel(channel=channel).plot(cmap="viridis")
        plt.title("Thermal Fronts (" + channel + ") - 0.05°, " + str(field.time.values)[:10])

        plt.show()

    def get_feature_map(self, field, channel=None):
        plt.figure(figsize=(10, 6))

        if channel is not None:
            field = field.sel(channel=channel)     # multi-channel stack -> pick one channel
            label = channel
        else:
            label = field.name or "field"          # e.g. "front_frequency"

        field.plot(cmap="viridis")
        plt.title(f"Thermal Fronts ({label}) - 0.05°, {str(field.time.values)[:10]}")
        plt.show()


tf = ThermalFronts(start_dt="2010-05-10T00:00:00", end_dt="2010-07-11T00:00:00")
tf.fetch_data()
thermal_field = tf.get_frequency_field(date="2010-06-15T00:00:00")
tf.get_feature_map(field = thermal_field)
