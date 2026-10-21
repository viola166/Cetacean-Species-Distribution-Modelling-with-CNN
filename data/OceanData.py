import copernicusmarine as cm
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass, field
import xarray as xr
import requests
from utils.sstToAnomaly import get_sst_anomaly
from utils.dataCropper import widen_bbox, widen_trange
from utils.sstToFront import get_front_channels, get_front_frequency
from utils.geoToEke import get_eke
from utils.caching import subset_path
from utils.swathData import regrid_to_common_grid, get_catalog_url, get_data_url
from utils.standardize import standardize_coords, standardize_longitude
from data.prepareGridData import to_target_grid
import pandas as pd

@dataclass(frozen=True)
class DatasetSpec:
    short_name: str             # useful for caching
    source: str                 # key into LOADERS: "cmems", "opendap", "ifremer_sst"
    location: str               # dataset_id or URL (or URL template)
    names: tuple                
    variables: tuple | None
    options: dict = field(default_factory=dict)   # source-specific extras

# One entry per kind of ocean data. Add new kinds here, nowhere else.
REGISTRY = {
    "ph": DatasetSpec(
        short_name="ph",
        source="cmems",
        location="cmems_mod_glo_bgc_my_0.25deg_P1M-m",  # ONLY MONTHLY VARIABLE!!!
        names=("Sea Water pH Reported on Total Scale",),
        variables=("ph",),
    ),
    "salinity": DatasetSpec(
        short_name="salinity",
        source="cmems",
        location="cmems_mod_glo_phy_my_0.083deg_P1D-m",  # verify with cm.describe()
        names=("Sea Water Salinity",),
        variables=("so",),
    ),
    "sst_low": DatasetSpec(
        short_name="sst_low",
        source="cmems",
        location="METOFFICE-GLO-SST-L4-REP-OBS-SST",
        names=("Foundation Sea Surface Temperature",),
        variables=("analysed_sst",),
    ),
    "geo_velocity": DatasetSpec(
        short_name="geo_velo",
        source="cmems",
        location="cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D",
        names=(
            "Absolut East-West Geostrophic Velocity", 
            "Absolut North-South Geostrophic Velocity", 
            "Anomaly East-West Geostrophic Velocity",
            "Anomaly North-South Geostrophic Velocity"
        ),
        variables=("ugos", "vgos", "ugosa", "vgosa"),
    ),
    "chlorophyll": DatasetSpec(
        short_name="chl",
        source="cmems",
        location="cmems_obs-oc_atl_bgc-plankton_my_l4-gapfree-multi-1km_P1D",
        names=("Mass Concentration of Chlorophyll-a in Sea Water",),
        variables=("CHL",),
    ),
    "oxygen": DatasetSpec(
        short_name="o2",
        source="cmems",
        location="cmems_mod_glo_bgc_my_0.25deg_P1D-m",
        names=("Mole Concentration of Dissolved Molecular Oxygen in Sea Water",),
        variables=("o2",),
    ),
    "wind": DatasetSpec(
        short_name="wind",
        source="cmems",
        location="cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H",
        names=("Sea Surface East-West Wind Speed", "Sea Surface North-South Wind Speed"),
        variables=("eastward_wind", "northward_wind"),
    ),
    "bathymetry": DatasetSpec(
        short_name="bathymetry",
        source="opendap",
        location="https://dap.ceda.ac.uk/"
                    "thredds/dodsC/bodc/gebco/global/gebco_2026/sub_ice_topography_bathymetry/netcdf/GEBCO_2026_sub_ice.nc",
        names=("Bathymetry",),
        variables=("elevation",),
    ),
    "fsle": DatasetSpec(
        short_name="fsle",
        source="opendap",
        location="http://tds-odatis.aviso.altimetry.fr/"
                "thredds/dodsC/dataset-duacs-dt-global-allsat-madt-fsle",
        names=("FSLE",),
        variables=("fsle_max",),
    ),
    "sst_anomaly": DatasetSpec(
        short_name="sst_a",
        source="cmems",
        location="METOFFICE-GLO-SST-L4-REP-OBS-SST",
        names=("Sea Surface Temperature Anomaly",),
        variables=("analysed_sst",),
        # define hyperparameters for climatological means here
        options={
            "clim_period": ("1981-10-01T00:00:00", "2026-03-31T00:00:00"),
            "function": "get_sst_anomaly", 
            "function_kwargs": {
                "frequency": "day",
            }
        }
    ),
    "thermal_fronts": DatasetSpec(
        short_name="thermal_fronts",
        source="cmems",
        location="METOFFICE-GLO-SST-L4-REP-OBS-SST",
        names=("Distance to Thermal Fronts", "Density of Thermal Fronts", "Gradient of Thermal Fronts"),
        variables=("analysed_sst",),
        options={
            "wide_bbox": (-27.5, -23.5, 35.8, 39.8),                       # necessary margin to compute CCA, always required (for standard bbox that is to small)
            "function" : "get_front_channels",
        }
    ),
    "thermal_front_freq": DatasetSpec(
        short_name="thermal_fronts",
        source="cmems",
        location="METOFFICE-GLO-SST-L4-REP-OBS-SST",
        names=("Frequency of Thermal Fronts",),
        variables=("analysed_sst",),
        options={
            "window": 30,
            "wide_bbox": (-27.5, -23.5, 35.8, 39.8),
            "function": "get_front_frequency",
            "function_kwargs": {
                "window": 30,
                "smooth_sigma": 1.0,
            }
        }
    ),
    "eddy_kinetic_e": DatasetSpec(
        short_name="eke",
        source="cmems",
        location="cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D",
        names=(
            "Eddy Kinetic Energy",
        ),
        variables=("ugos", "vgos"),
        # define hyperparameters for subset and climatological means here
        options={
            "clim_period": ("1993-01-01T00:00:00", "2026-01-16T00:00:00"),
            "function": "get_eke", 
            "function_kwargs": {
                "frequency": "day", 
            }
        }
    ),
    "sst_high": DatasetSpec(
        short_name="sst_high",
        source="ifremer_sst",
        location="https://tds0.ifremer.fr/thredds/catalog/",
        names=("Merged High-Resolution SST composite",),
        variables=None,
    )

}

# ---------- loaders: (spec, bbox, time_range) -> xr.Dataset ----------

def load_cmems(spec, bbox, trange, refresh):
    """Load a dataset from Copernicus Marine, crop to bbox and trange, cache to disk."""
    path = subset_path(spec=spec, bbox=bbox, trange=trange, name="raw")
    if path.exists() and refresh is False:
        return xr.load_dataset(path)

    lon0, lon1, lat0, lat1 = bbox

    ds = cm.open_dataset(
        dataset_id=spec.location,
        variables=list(spec.variables),
        minimum_longitude=lon0, maximum_longitude=lon1,
        minimum_latitude=lat0, maximum_latitude=lat1,
        start_datetime=trange[0], end_datetime=trange[1],
    ).load()  # actually download now, not lazily
    ds = standardize_coords(ds)  # ensure consistent lat/lon names and order
    ds = standardize_longitude(ds)  # ensure longitudes are in -180..180
    if "depth" in ds.dims:
        ds = ds.isel(depth=0) 
    ds = to_target_grid(ds)
    ds.to_netcdf(path)
    return ds


def load_opendap(spec, bbox, trange, refresh):
    """Load a dataset from an OPeNDAP URL, crop to bbox and trange, cache to disk."""
    path = subset_path(spec=spec, bbox=bbox, trange=trange, name="raw")
    if path.exists() and refresh is False:
        return xr.load_dataset(path)

    # access complete dataset
    session = requests.Session()
    ds = xr.open_dataset(
        spec.location,          # lazy, OPeNDAP
        engine="pydap",
        session=session
    )
    ds = standardize_coords(ds)  # ensure consistent lat/lon names and order
    ds = standardize_longitude(ds)  # ensure longitudes are in -180..180
    lon0, lon1, lat0, lat1 = bbox
    # match the dataset's longitude convention (0-360 vs -180-180)
    if float(ds.longitude.max()) > 180:
        lon0, lon1 = lon0+360, lon1+360
    print("bbox: ", lon0, lon1, lat0, lat1)
    ds = ds.sortby("latitude").sortby("longitude")             # slices need ascending order
    # get subset
    ds = ds.sel(
        longitude=slice(lon0, lon1), 
        latitude=slice(lat0, lat1)
    )
    if "time" in ds.dims:                           # bathymetry has no time axis
        ds = ds.sel(time=slice(*trange))
    if "depth" in ds.dims:
        ds = ds.isel(depth=0) 

    ds = to_target_grid(ds)

    subset = ds[list(spec.variables)]
    subset.to_netcdf(path)
    return subset


def load_ifremer_sst(spec, bbox, trange, refresh):
    """Load IFREMER SST swath data, regrid to a common grid, cache to disk."""
    start, end = trange
    days = pd.date_range(start, end, freq="D")  # inclusive on both ends
    session = requests.Session()
    daily = []

    for d in days:
        day = d.strftime("%Y-%m-%d")
        path = subset_path(spec=spec, bbox=bbox, trange=(day, day), name="raw")

        # 1) cached day
        if path.exists():
            daily.append(xr.load_dataset(path))
            continue
        # 2) download + regrid
        ds_one_day = []
        try:
            catalog = get_catalog_url(d)
            data_urls = get_data_url(bbox, catalog)

            ds_one_day = [
                xr.open_dataset(url, engine="pydap", session=session)
                for url in data_urls
            ]
            if not ds_one_day:
                print(f"Skipping {d:%Y-%m-%d}: no datasets found")
                continue
            ds_one_day = [standardize_coords(ds) for ds in ds_one_day]
            ds_one_day = [standardize_longitude(ds) for ds in ds_one_day]
            day_ds = regrid_to_common_grid(ds_one_day, bbox, spec)

        except Exception as e:  # missing catalog, 404, network error, ...
            print(f"Skipping {d:%Y-%m-%d}: {e}")
            continue
        finally:
            for ds in ds_one_day:
                ds.close()

        day_ds = day_ds.expand_dims(time=[pd.Timestamp(day)])
        day_ds = day_ds.sortby("latitude").sortby("longitude")
        if spec.variables is not None:
            day_ds = day_ds[list(spec.variables)]

        day_ds.to_netcdf(path)
        daily.append(day_ds)

    if not daily:
        raise ValueError(f"No IFREMER data found for {start} to {end}")

    return xr.concat(daily, dim="time").sortby("time")


def load_any(spec, bbox, trange, refresh):
    """Load a dataset from any source, crop to bbox and trange, cache to disk."""
    # (keep the user's requested ranges for cropping afterwards)
    user_bbox, user_trange = bbox, trange

    # add time window for thermal front frequency
    time_offset = spec.options.get("window")

    # override bbox and trange if REGISTRY options are broader
    bbox = widen_bbox(bbox, spec.options.get("wide_bbox"))
    trange = widen_trange(trange, spec.options.get("clim_period"), time_offset)

    ds = LOADERS[spec.source](spec, bbox, trange, refresh)
    # apply any post-processing function (e.g. climatology, anomaly, EKE, thermal fronts)
    fn_name = spec.options.get("function")
    if fn_name is not None:
        if fn_name not in PROCESSORS:
            raise KeyError(f"Unknown function {fn_name!r}. Available: {sorted(PROCESSORS)}")
        kwargs = dict(spec.options.get("function_kwargs", {}))
        kwargs["user_trange"] = user_trange
        kwargs["user_bbox"] = user_bbox
        kwargs["spec"] = spec
        if trange != user_trange:
            kwargs["full_trange"] = trange
        if bbox != user_bbox:
            kwargs["wider_bbox"] = bbox
        ds = PROCESSORS[fn_name](ds, **kwargs)

    print("got", spec.short_name)

    return ds


LOADERS = {
    "cmems": load_cmems,
    "opendap": load_opendap,
    "ifremer_sst": load_ifremer_sst,
}

PROCESSORS = {
    "get_eke": get_eke,
    "get_front_channels": get_front_channels,
    "get_sst_anomaly": get_sst_anomaly,
    "get_front_frequency": get_front_frequency,
}



# ---------------------class to access all data---------------------

class OceanData:
    def __init__(
        self,
        kinds,
        minimum_longitude=-26.5,
        maximum_longitude=-24.5,
        minimum_latitude=37,
        maximum_latitude=38.5,
        start_dt="2008-05-01T00:00:00",         # extend of available csv data
        end_dt="2018-12-31T00:00:00",
        # cache_dir="output",
        refresh = False,
    ):
        if isinstance(kinds, str):
            kinds = [kinds]
        self.kinds = [k.lower() for k in kinds]

        if kinds == ["all"]:
            self.kinds = list(REGISTRY.keys())
        else:
            self.kinds = list(kinds)

        unknown = [k for k in self.kinds if k not in REGISTRY]
        if unknown:
            raise ValueError(f"Unknown kind(s) {unknown}. Available: {sorted(REGISTRY)}")

        self.specs = {k: REGISTRY[k] for k in self.kinds}
        self.min_lon, self.max_lon = minimum_longitude, maximum_longitude
        self.min_lat, self.max_lat = minimum_latitude, maximum_latitude
        self.start_dt, self.end_dt = start_dt, end_dt
        self.refresh = refresh
        self._ds = {}   # kind -> xarray.Dataset, filled lazily
  
    def fetch_data(self):
        """Open every requested dataset (cached). Returns {kind: Dataset}."""
        bbox = (self.min_lon, self.max_lon, self.min_lat, self.max_lat)
        trange = (self.start_dt, self.end_dt)
        for kind, spec in self.specs.items():
            if kind not in self._ds:
                self._ds[kind] = load_any(spec, bbox, trange, self.refresh)
        return self._ds

    def insert_data(self, kind, ds):
        """Insert a dataset for a kind, replacing any existing one."""
        if kind not in self.specs:
            raise ValueError(f"Unknown kind {kind}. Available: {sorted(self.specs)}")
        self._ds[kind] = ds

    def resolution(self, kind):
        """Spatial resolution (lon, lat) in degrees, read from the data itself."""
        ds = self.fetch_data()[kind]
        dlon = float(np.abs(np.diff(ds.longitude.values)).mean())
        dlat = float(np.abs(np.diff(ds.latitude.values)).mean())
        return dlon, dlat


    def get_slice(self, kind, depth_index=0, time_index=0, var=None):
        """DataArray for the kind's variable, surface by default, with unified
        latitude/longitude coords (longitude in -180..180)."""
        ds = self.fetch_data()[kind]
        if isinstance(ds, xr.DataArray):                  # safety net for stragglers
            ds = ds.to_dataset()

        if "depth" in ds.dims:
            ds = ds.isel(depth=depth_index)
        if "time" in ds.dims:
            ds = ds.isel(time=time_index)
        if var is None:
            return ds
        if isinstance(var, str):
            var = [var]
        return ds[list(var)]
    

    def get_feature_map(self, kind, time_index=0, depth_index=0, var=None):
        spec = self.specs[kind]
        ds = self.get_slice(kind, depth_index=depth_index, time_index=time_index, var=var)
        dlon, dlat = self.resolution(kind)

        for i, (name, da) in enumerate(ds.data_vars.items()):
            field = da.transpose("latitude", "longitude")
            label = spec.names[i] if len(spec.names) == len(ds.data_vars) else name

            fig, ax = plt.subplots(figsize=(10, 6))
            pc = ax.pcolormesh(field["longitude"], field["latitude"], field,
                            cmap="viridis", shading="auto")
            plt.colorbar(pc, ax=ax, label=name)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")

            title = f"{label} - {dlon:.3g}°×{dlat:.3g}°"
            if "time" in field.coords:
                title += f", {str(field['time'].values)[:10]}"
            ax.set_title(title)

        plt.show()
   

# data_instance = OceanData(kinds=["all"], start_dt="2010-06-01T00:00:00", end_dt="2010-06-02T00:00:00")
# data = data_instance.fetch_data()
# for dataset in data.values():
#     print(dataset)

# data_instance.get_feature_map(kind="thermal_fronts", time_index=0)