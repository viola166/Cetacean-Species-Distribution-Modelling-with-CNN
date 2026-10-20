from torch.utils.data import DataLoader
from data.prepareGridData import remove_skewness, normalize_channel
from data.prepareCSVData import build_samples
from data.OccurrenceDataset import OccurrenceDataset
from data.prepareDataset import preprocess_and_build
from data.prepareCSVData import load_csv_data
from data.DataSplitter import DataSplitter
from data.OceanData import OceanData
from pathlib import Path


def preprocess_and_build(csv_data, split, channel_dss, incl_mask_channels=False, time_res="D", batch_size=256):
    """
    Build everything needed to train and evaluate on ONE cross-validation fold:
    preprocessed channels, labelled datasets (patches + labels + weights) and DataLoaders.

    csv_data     : dict {key: record} with all occurrence records
                   (record: scientificName, eventDate, decimalLatitude, decimalLongitude, ...)
    split         : one entry of DataSplitter.split(csv_data), a dict with the keys
                   "fold", "train", "val", "test" (lists of record keys)
    species_list : the ~10 target species; fixes the column order of Y (and of the model outputs)
    channel_dss  : dict {channel_name: RAW channel Dataset} on the common grid
                   (one or more variables each).
    time_res     : finest time resolution among the channels ("D" daily, "h" hourly, ...);
                   records of the same grid cell and time bin are merged into one sample
    batch_size   : batch size of all DataLoaders. Use a large one (256+), so that rare species
                   have positives in most batches (needed for the nnPU risk terms).

    Returns
    -------
    data    : dict {"train", "val", "test"} -> OccurrenceDataset (splits without records are absent)
              each dataset has
                .X     (N, 2C, P, P) float32, normalized patches: C value channels + C validity
                       masks (1 = valid, 0 = land/missing/off-grid), P = NUMBER_OF_PIXELS_PER_PATCH
                .Y     (N, S) float32, multi-hot labels: 1 = species recorded in this sample,
                       0 = unlabelled (not recorded; may still be present)
                .W     (N, S) float32, de-clustering weights, mean 1, the same for all species
                       of a sample
                .meta  DataFrame (latitude, longitude, time), one row per sample
                       (cell-centre coordinates), aligned with X, Y and W
    loaders : dict with the same keys -> DataLoader yielding batches (x, y, w).
              train: shuffled, last incomplete batch dropped; val/test: ordered, complete.
    pi      : (S,) tensor, class prior per species for nnPU. It is the weighted share of
              training samples where the species was recorded, clamped to [1e-3, 1 - 1e-3].
    """

    # ---- 1. Preprocess each channel dataset ----
    norm_dss, mask_dss = {}, {}
    for name, ds in channel_dss.items():
        ds_t, _ = remove_skewness(ds, csv_data, split, verbose=False)        # decided on train records
        n, m, _ = normalize_channel(ds_t, csv_data, split, verbose=False)    # fitted on train records
        norm_dss[name] = n
        mask_dss[name] = m

    if not incl_mask_channels:
        mask_dss_list = None
    else:
        mask_dss_list = list(mask_dss.values())

    # ---- 2. Labels, weights and sample coordinates (only the extents of the raw datasets are used) ----
    samples = build_samples(csv_data, split, channel_dss, time_res=time_res)

    # ---- 3. One patch per sample; values are taken in the same (dict) order for norm and mask ----
    data = {name: OccurrenceDataset(Y, W, meta, list(norm_dss.values()), mask_dss_list)
            for name, (Y, W, meta) in samples.items()}

    # ---- 4. Weighted class prior for nnPU ----
    Y_tr, W_tr = data["train"].Y, data["train"].W
    pi = ((W_tr * Y_tr).sum(0) / W_tr.sum(0)).clamp(1e-3, 1 - 1e-3)

    # ---- 5. DataLoaders ----
    loaders = {
        "train": DataLoader(data["train"], batch_size=batch_size, shuffle=True, drop_last=True),
        **{k: DataLoader(v, batch_size=batch_size, shuffle=False)
           for k, v in data.items() if k != "train"},
    }
    return data, loaders, pi


def prepare_data(self, csv_data=None, channel_dss=None, split_csv_data_keys=None, incl_mask_channels=False):
        # get CSV data
        if csv_data is None:
            csv_data = load_csv_data()

        # get data split
        if split_csv_data_keys is None:
            data_splitter = DataSplitter()
            split_csv_data_keys = data_splitter.spatial_block_cv(csv_data=csv_data)

        # get grid data
        if channel_dss is None:
            grid_data_Accessor = OceanData(kinds=["all"])
            channel_dss = grid_data_Accessor.fetch_data() # dict of all channel xarray Datasets

        # preprocess data in bring in correct format
        data, loaders, pi = preprocess_and_build(csv_data, split_csv_data_keys[0], channel_dss, incl_mask_channels)     # for now only training the first split

        return data, loaders, pi


def get_num_in_channels(occurrence_ds):
    x0, _, _ = occurrence_ds["train"][0]   
    return x0.shape[0]
     