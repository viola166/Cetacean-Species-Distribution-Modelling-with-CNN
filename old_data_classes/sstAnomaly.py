from data.sst_low_res import SST_LowRes
import matplotlib.pylab as plt
import numpy as np
import xarray as xr



class SSTAnomaly(SST_LowRes):
    def __init__(self, *args, **kwargs):
        kwargs["start_dt"] = "1981-10-01T00:00:00"   # always the full record for climatological means
        kwargs["end_dt"] = "2026-03-31T00:00:00"
        super().__init__(*args, **kwargs)

    def fetch_data(self, refresh=False):
        """SST anomaly relative to the daily climatology (DataArray)"""
        path = self._path("sstAnomaly")
        if path.exists() and not refresh:
            return xr.load_dataarray(path)

        ds = super().fetch_data()
        clim = self.get_climatology()
        sst_anomaly = ds.analysed_sst.groupby("time.dayofyear") - clim.analysed_sst
        sst_anomaly.to_netcdf(path)
        return sst_anomaly

    def get_field(self, date="2020-01-01T00:00:00"):
        ds = self.fetch_data()
        field_tuple = (ds.sel(time=date, method="nearest"), "Sea Surface Temperature Anamoly")
        return field_tuple

    def get_feature_map(self, field_tuple):
        super().get_feature_map(field_tuple)


sstA = SSTAnomaly()
sstA.fetch_data()
sst_anomaly_field = sstA.get_field()
sstA.get_feature_map(sst_anomaly_field)

    

    

    

