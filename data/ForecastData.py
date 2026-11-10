from data.OceanData import DatasetSpec, OceanData
from datetime import date


class ForecastData(OceanData):
    ### Adapted registry with forcast dataset locations ###
    REGISTRY = {
        "ph": DatasetSpec(
            short_name="ph",
            source="cmems",
            location="cmems_mod_glo_bgc-car_anfc_0.25deg_P1D-m",
            names=("Sea Water pH Reported on Total Scale",),
            variables=("ph",),
        ),
        "salinity": DatasetSpec(
            short_name="salinity",
            source="cmems",
            location="cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m",  # verify with cm.describe()
            names=("Sea Water Salinity",),
            variables=("so",),
        ),
        "sst_low": DatasetSpec(
            short_name="sst_low",
            source="cmems",
            location="METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2",
            names=("Foundation Sea Surface Temperature",),
            variables=("analysed_sst",),
        ),
        "geo_velocity": DatasetSpec(
            short_name="geo_velo",
            source="cmems",
            location="cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.125deg_P1D",
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
            location="cmems_obs-oc_atl_bgc-plankton_my_l4-gapfree-multi-1km_P1D", #"cmems_obs-oc_atl_bgc-plankton_nrt_l4-gapfree-multi-1km_P1D",
            names=("Mass Concentration of Chlorophyll-a in Sea Water",),
            variables=("CHL",),
            options={
                "time_offset": 85,
                "train_spec": OceanData.REGISTRY["chlorophyll"],
            }
        ),
        "oxygen": DatasetSpec(
            short_name="o2",
            source="cmems",
            location="cmems_mod_glo_bgc-bio_anfc_0.25deg_P1D-m",
            names=("Mole Concentration of Dissolved Molecular Oxygen in Sea Water",),
            variables=("o2",),
        ),
        "wind": DatasetSpec(
            short_name="wind",
            source="cmems",
            location="cmems_obs-wind_glo_phy_nrt_l4_0.125deg_PT1H",
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
            location="METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2",
            names=("Sea Surface Temperature Anomaly",),
            variables=("analysed_sst",),
            # define hyperparameters for climatological means here
            options={
                "clim_period": ("1981-10-01T00:00:00", "2026-03-31T00:00:00"),
                "function": "get_sst_anomaly", 
                "function_kwargs": {
                    "frequency": "day",
                    "train_spec": OceanData.REGISTRY["sst_anomaly"],
                }
            }
        ),
        "thermal_fronts": DatasetSpec(
            short_name="thermal_fronts",
            source="cmems",
            location="METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2",
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
            location="METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2",
            names=("Frequency of Thermal Fronts",),
            variables=("analysed_sst",),
            options={
                "time_offset": 30,
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
            location="cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.125deg_P1D",
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
                    "train_spec": OceanData.REGISTRY["eddy_kinetic_e"], 
                }
            }
        ),
        "sst_high": DatasetSpec(
            short_name="sst_high",
            source="ifremer_sst",
            location="https://tds0.ifremer.fr/thredds/catalog/",
            names=("Merged High-Resolution SST composite",),
            variables=None,
        ),
        "dist_to_coast": DatasetSpec(
            short_name="d_coast",
            source="opendap",
            location="https://dap.ceda.ac.uk/"
                        "thredds/dodsC/bodc/gebco/global/gebco_2026/sub_ice_topography_bathymetry/netcdf/GEBCO_2026_sub_ice.nc",
            names=("Distance to Coast",),
            variables=("elevation",),
            options={
                "function": "get_dist_to_coast",
            }
        )
    }

    def __init__(self, kinds, start_dt=date.today().isoformat(), end_dt=date.today().isoformat(), **kwargs):
        super().__init__(kinds, start_dt=start_dt, end_dt=end_dt, **kwargs)
        # forecast-specific setup here, after the base class has built self.specs

      

