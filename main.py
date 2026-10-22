from cnn.model import SpeciesCNN
from data.prepareDataset import prepare_data, get_num_in_channels
from data.OceanData import OceanData
from pathlib import Path
import json
import xarray as xr

# for testing purposes, only use fraction of the dataset
grid_data_accessor = OceanData(kinds=["salinity", "eddy_kinetic_e", "sst_low", "geo_velocity", "chlorophyll", "sst_anomaly", "oxygen"], refresh = False)
channel_dss = grid_data_accessor.fetch_data()
for kind, ds in channel_dss.items():
    print(kind)
    for var in ds.data_vars:
        print(f"  {var}: {ds[var].dims} {ds[var].shape}")

data, loaders, pi = prepare_data(channel_dss=channel_dss)
in_channels = get_num_in_channels(data)

model = SpeciesCNN(in_channels=in_channels)

run_dir = Path("runs/nnpu_fold_0")
run_dir.mkdir(parents=True, exist_ok=True)

model.train_sdm(loaders, pi, run_dir=run_dir)

test_auc = model.evaluate_sdm(loaders["test"])          # evaluate once, after model selection
with open(run_dir / "test_metrics.json", "w") as f:
    json.dump(test_auc, f, indent=2)
