from old_data_classes.geoVelocity import GeoVelocity
import xarray as xr

class EddyKineticEnergy(GeoVelocity):
    def __init__(self, *args, **kwargs):
        kwargs["start_dt"] = "1993-01-01T00:00:00"
        kwargs["end_dt"] = "2026-01-16T00:00:00"
        super().__init__(*args, **kwargs)

    ### Put methods from GeoVelocity here, so functions that are not specific to geostrophic velocity 
    ### are available in EddyKineticEnergy instead.

    def fetch_data(self, refresh=False):
        """Eddy kinetic energy from anomalies relative to the daily climatology."""
        path = self._path("eke")
        if path.exists() and not refresh:
            return xr.load_dataarray(path)

        ds = super().fetch_data()
        clim = self.get_climatology()
        u_anom = ds.ugos.groupby("time.dayofyear") - clim.ugos
        v_anom = ds.vgos.groupby("time.dayofyear") - clim.vgos
        eke = (0.5 * (u_anom**2 + v_anom**2)).rename("eke")
        eke.to_netcdf(path)
        return eke

    def get_field(self, date):
        eke = self.fetch_data()
        # caution: sel/ not isel > not selecting index but exact date/value
        field = eke.sel(time=date, method="nearest")
        return field

    def get_feature_map(self, field):
        """Map of EKE for a single day, e.g. get_feature_map("2020-01-01")."""
        super().get_feature_map([(field, "Eddy Kinetic Energy")], label="EKE (m²/s²)", cmap="magma")


    def get_climatology(self, refresh=False):
        """Daily climatological mean (by day of year) of ugos and vgos."""
        path = self._path("clim")
        if path.exists() and not refresh:
            return xr.load_dataset(path)

        ds = super().fetch_data()  # raw GeoVelocity data, NOT self.fetch_data()
        clim = ds.groupby("time.dayofyear").mean(dim="time")
        clim.to_netcdf(path)
        return clim


    
eke_instance = EddyKineticEnergy()
eke_one_date = eke_instance.get_field("2025-01-16T00:00:00")
eke_instance.get_feature_map(eke_one_date)
