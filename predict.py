from cnn.model import SpeciesCNN
import torch
from data.ForecastData import ForecastData
from data.prepareDataset import prepare_data, get_num_in_channels
from data.prepareGridData import preprocess_channel, make_patches
from datetime import date

#-------random location for prediction--------
LAT = 37.81
LON = -24.94
FORECAST_TIME = date.today()
#---------------------------------------------

model, meta = SpeciesCNN.load_run(run_dir="runs/nnpu_fold_7")
pre = meta["preprocessing"]
transforms = pre["transforms"]
norm_stats = {v: tuple(ms) for v, ms in pre["stats"].items()}   # JSON gives lists

grid_data_accessor = ForecastData(kinds=["all"], refresh=True)
channel_dss = grid_data_accessor.fetch_data()    # {kind: Dataset}

# norm_dss, mask_dss = [], []
# for kind, ds in channel_dss.items():
#     norm_ds, mask_ds = preprocess_channel(ds, transforms, norm_stats, eps=pre["eps"])
#     norm_dss.append(norm_ds)
#     mask_dss.append(mask_ds)

#     print(kind)
#     for var in norm_ds.data_vars:
#         print(f"  {var}: {norm_ds[var].dims} {norm_ds[var].shape}")


# x = make_patches(norm_dss, mask_dss, LAT, LON, FORECAST_TIME)    # (C, 13, 13)
# assert x.shape[0] == meta["in_channels"]
# probs = model(torch.from_numpy(x)[None])                          # (1, n_species)

norm_dss, mask_dss = [], []
for name in pre["channel_order"]:                # stored order, not dict order
    ds = channel_dss[name]

    # same steps as training: keep only trained variables
    stats = {v: tuple(ms) for v, ms in pre["stats"][name].items()}
    ds = ds[list(stats)]

    norm_ds, mask_ds = preprocess_channel(ds, pre["transforms"][name], stats, eps=pre["skew_eps"])
    norm_dss.append(norm_ds)
    mask_dss.append(mask_ds)

    print(name)
    for var in norm_ds.data_vars:
        print(f"  {var}: {norm_ds[var].dims} {norm_ds[var].shape}")

mask_list = mask_dss if pre["use_mask"] else None
x = make_patches(norm_dss, mask_list, LAT, LON, FORECAST_TIME)    # (C, 13, 13)
assert x.shape[0] == meta["in_channels"]

with torch.no_grad():
    probs = model(torch.from_numpy(x)[None])[0]                   # (n_species,)

for sp, p in zip(meta["species_list"], probs.tolist()):
    print(f"{sp}: {p:.3f}")