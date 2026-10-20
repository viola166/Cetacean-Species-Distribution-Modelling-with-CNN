from data.prepareCSVData import load_csv_data
from data.prepareGridData import to_target_grid, ds_to_tensor, normalize_channel, histogram, remove_skewness
from data.OceanData import OceanData
from matplotlib import pyplot as plot
import pandas as pd
from data.DataSplitter import DataSplitter

### TEST OCENDATA FETCHING AND REGRIDDING ###
# # Initilaize data accessor and fetch data
# dataAccess = OceanData(kinds=["sst_high"], start_dt="2010-06-01T00:00:00", end_dt="2010-06-05T00:00:00")
# data = dataAccess.fetch_data()

# # show the fetched dataset before regridding
# dataAccess.get_feature_map(kind="sst_high", time_index=0)

# # get sst_high dataset and regrid it to the target grid
# sst_high_ds = data["sst_high"]
# sst_high_ds_common_grid = to_target_grid(sst_high_ds, TARGET_GRID)

# # insert the regridded dataset back into the data dictionary/Oceandata instance
# dataAccess.insert_data(kind="sst_high", ds=sst_high_ds_common_grid)

# # show the regridded dataset
# dataAccess.get_feature_map(kind="sst_high", time_index=0)

# print("SST High dataset after regridding:", sst_high_ds_common_grid)
# sst_high_tensor = ds_to_tensor(sst_high_ds_common_grid)
# print("SST High tensor shape:", sst_high_tensor.shape)  # Should be (N, C, H, W)


dataAccess = OceanData(kinds=["chlorophyll"], start_dt="2009-12-01T00:00:00", end_dt="2010-12-01T00:00:00")
data = dataAccess.fetch_data()

# show the fetched dataset before regridding
dataAccess.get_feature_map(kind="chlorophyll", time_index=0)

# get chlorophyll dataset and regrid it to the target grid
chlorophyll_ds = data["chlorophyll"]
print(chlorophyll_ds)
chlorophyll_ds_common_grid = to_target_grid(chlorophyll_ds)
print("Chlorophyll after regridding: ", chlorophyll_ds_common_grid)

# insert the regridded and transformed dataset back into the data dictionary/Oceandata instance
dataAccess.insert_data(kind="chlorophyll", ds=chlorophyll_ds_common_grid)

# show the regridded dataset
dataAccess.get_feature_map(kind="chlorophyll", time_index=0)

### TEST TILE ASSIGNMENT AND DATA SPLITTING ###
csv_data = load_csv_data()

data_splitter = DataSplitter()

# NOT NECESSARY: just for visualization, explicitly assign tiles to the target data
tile_id = data_splitter.assign_tiles_to_pixels(n_pixels=20, spatial_buffer=1)
tile_id.plot()
plot.show()

folds = data_splitter.split(csv_data) # internally calls assign_tiles_to_targets() to assign each target to a tile, then splits the tiles into train/val/test sets

for fold in folds:
    train = {k: csv_data[k] for k in fold["train"]}
    val   = {k: csv_data[k] for k in fold["val"]}
    test  = {k: csv_data[k] for k in fold["test"]}

    # check for potentially skewed variables and remove skewness
    unskewed_chl_ds, transforms = remove_skewness(channel_ds=chlorophyll_ds_common_grid, csv_data=csv_data, fold=fold)

    # normalize all channel data
    norm_ds, mask_ds, norm_stats = normalize_channel(channel_ds=chlorophyll_ds_common_grid, csv_data=csv_data, fold=fold)
    norm_ds_log, mask_ds_log, norm_stats_log = normalize_channel(channel_ds=unskewed_chl_ds, csv_data=csv_data, fold=fold)

    norm_for_plot = norm_ds_log.where(mask_ds_log.rename({f"{v}_mask": v for v in norm_ds.data_vars}) > 0)
    histogram(channel_ds=norm_for_plot, csv_data=csv_data, fold=fold)

    print(fold["fold"], len(train), len(val), len(test))
    # fit normalization on `train` only, train the CNN, evaluate on `val`, then on `test`