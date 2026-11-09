from cnn.model import SpeciesCNN
from data.prepareDataset import prepare_data, get_num_in_channels
from data.prepareGridData import NUMBER_OF_PIXELS_PER_PATCH
from data.OceanData import OceanData
from data.ForecastData import ForecastData
from pathlib import Path
import json
import xarray as xr
import pandas as pd

# for testing purposes, only use fraction of the dataset
# grid_data_accessor = OceanData(kinds=["ph", "salinity", "sst_low", "chlorophyll", "geo_velocity", "oxygen", "wind", "bathymetry", "dist_to_coast", "fsle", "sst_anomaly", "eddy_kinetic_e", "thermal_fronts", "thermal_front_freq"], refresh = False)
kinds = ["ph", "salinity", "sst_low", "geo_velocity", "chlorophyll", "oxygen", "wind", "bathymetry", "fsle", "sst_anomaly", "thermal_fronts", "thermal_front_freq", "eddy_kinetic_e", "dist_to_coast"]
grid_data_accessor = OceanData(kinds=kinds)
channel_dss = grid_data_accessor.fetch_data()
for kind, ds in channel_dss.items():
    print(kind)
    grid_data_accessor.get_feature_map(kind=kind, time_index=28)
    for var in ds.data_vars:
        print(f"  {var}: {ds[var].dims} {ds[var].shape}")

skew_eps=1e-3
data, loaders, pi, transforms, n_stats = prepare_data(channel_dss=channel_dss, skew_eps=skew_eps)
in_channels = get_num_in_channels(data)

model = SpeciesCNN(in_channels=in_channels)

run_dir = Path("runs/nnpu_fold_7")
run_dir.mkdir(parents=True, exist_ok=True)

model.preproc = {
    "transforms": transforms,                                   # {var: kind} from remove_skewness
    "stats": {name: {v: [float(m), float(s)] for v, (m, s) in d.items()}
                for name, d in n_stats.items()},
    "patch_size": NUMBER_OF_PIXELS_PER_PATCH,
    "chl_lead_weeks": 12,
    "channel_order": list(n_stats.keys()),      # variable names in the exact order make_patches stacks them
    "use_mask": False,
    "skew_eps": skew_eps,
}

model.train_sdm(loaders, pi, run_dir=run_dir)

test_auc = model.evaluate_sdm(loaders["test"])          # evaluate once, after model selection
with open(run_dir / "test_metrics.json", "w") as f:
    json.dump(test_auc, f, indent=2)
