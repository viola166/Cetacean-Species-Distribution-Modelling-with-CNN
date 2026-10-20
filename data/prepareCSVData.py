from pathlib import Path
import csv
import pandas as pd
import warnings
import numpy as np
from data.prepareGridData import _valid_records, extract_patch, TARGET_GRID, NUMBER_OF_PIXELS_PER_PATCH
from scipy.spatial import cKDTree


EVENT_TABLE_NAME = "event.txt"
OCCURRENCE_TABLE_NAME = "occurrence.txt"
SPECIES_LIST = [
    "Balaenoptera musculus",
    "Physeter macrocephalus",
    "Balaenoptera physalus",
    "Balaenoptera borealis",
    "Megaptera novaeangliae",
    "Stenella Frontalis",
    "Stenella coeruleoalba",
    "Tursiops truncatus",
    "Delphinus delphis",
    "Grampus griseus",
]
N_SPECIES = len(SPECIES_LIST)


def load_csv_data(name_event_table=EVENT_TABLE_NAME, name_occurrence_table=OCCURRENCE_TABLE_NAME):
    """
    Load target data from the event and occurrence tables, and return a dictionary of targets.
    the keys of the dictionary are the eventIDs.

    Each unique (eventID, scientificName) pair in the occurrence table gives one datapoint.
    The first species of an event (alphabetical) keeps the original eventID; further species
    at the same event get derived IDs "<eventID>_2", "<eventID>_3", ...
    Time and location come from the event table, matched on the original eventID.

    Returns
    -------
    dict {key: {"originalEventID", "scientificName", "eventDate",
                "decimalLatitude", "decimalLongitude"}}
    """
    target_dir = "targets"
    event_path = Path(target_dir, name_event_table)
    occurrence_path = Path(target_dir, name_occurrence_table)
    
    read_kw = dict(sep="\t", dtype=str, quoting=csv.QUOTE_NONE, encoding="utf-8-sig")

    occ = pd.read_csv(occurrence_path, usecols=["eventID", "scientificName"], **read_kw)
    ev = pd.read_csv(
        event_path,
        usecols=["eventID", "eventDate", "decimalLatitude", "decimalLongitude"],
        **read_kw,
    )

    # ---- occurrence table: one row per unique (eventID, species) ----
    for col in occ.columns:
        occ[col] = occ[col].str.strip().replace("", pd.NA)
    occ = occ.dropna(subset=["eventID"])                    # drop rows with missing eventID, as they cannot be matched to the event table      
    n_events_occ = occ["eventID"].nunique()                 # number of unique eventIDs in the occurrence table

    occ = occ.dropna(subset=["scientificName"]).drop_duplicates()       # drop rows with missing scientificName, as they cannot be used as targets, drop duplicates of (eventID, scientificName) pairs
    n_lost = n_events_occ - occ["eventID"].nunique()
    if n_lost:
        warnings.warn(f"{n_lost} eventIDs have no scientificName and were dropped.")

    occ = occ.sort_values(["eventID", "scientificName"], kind="stable")
    rank = occ.groupby("eventID").cumcount()                       # 0, 1, 2, ... per event
    occ["originalEventID"] = occ["eventID"]
    occ["key"] = occ["eventID"].where(rank == 0, occ["eventID"] + "_" + (rank + 1).astype(str))

    # ---- event table: one row per eventID with date and location ----
    for col in ev.columns:
        ev[col] = ev[col].str.strip().replace("", pd.NA)
        ev = ev.dropna(subset=["eventID"]).drop_duplicates("eventID", keep="first")         

    # Darwin Core dates can be ranges ("2010-06-01/2010-06-02"): use the start
    start = ev["eventDate"].str.split("/").str[0]
    ev["eventDate"] = pd.to_datetime(start, format="ISO8601", errors="coerce", utc=True).dt.tz_localize(None)
    ev["decimalLatitude"] = pd.to_numeric(ev["decimalLatitude"], errors="coerce")
    ev["decimalLongitude"] = pd.to_numeric(ev["decimalLongitude"], errors="coerce")
    bad_coord = ~ev["decimalLatitude"].between(-90, 90) | ~ev["decimalLongitude"].between(-180, 180)
    ev.loc[bad_coord, ["decimalLatitude", "decimalLongitude"]] = float("nan")

    # ---- join: only events listed in occurrence.txt are kept ----
    df = occ.merge(ev, on="eventID", how="left")
    needed = ["eventDate", "decimalLatitude", "decimalLongitude"]
    incomplete = df[needed].isna().any(axis=1)
    if incomplete.any():
        n_ids = df.loc[incomplete, "originalEventID"].nunique()
        warnings.warn(
            f"{n_ids} eventIDs have no match in the event table or lack a valid date/"
            f"coordinates; {int(incomplete.sum())} datapoints dropped."
        )
        df = df[~incomplete]

    if not df["key"].is_unique:
        raise ValueError("Derived eventIDs are not unique; an ID like '<eventID>_2' already exists.")

    df = df.set_index("key")[["originalEventID", "scientificName"] + needed]
    return df.to_dict("index")


def compute_weights(Y_df, n_species=N_SPECIES, radius_px=3, time_days=2, power=0.5, max_ratio=1.5):
    """
    max_ratio : largest allowed ratio between the highest and lowest weight.
                1.0 = all weights equal, 1.5-2.0 = mild de-clustering.
    meta      : DataFrame with columns 'i', 'j' (spatial pixels), and 'time' (datetime).
    n_species : Number of species columns to replicate the weights across.
    radius_px : Spatial bandwidth in pixels for neighborhood scaling.
    time_days : Temporal bandwidth in days for neighborhood scaling.
    power     : Exponent for inverse-density weighting (w = n^-power). Higher = stronger penalty on clusters.
    max_ratio : Max allowed ratio between highest and lowest weight (floor). 1.0 = uniform, 1.5-2.0 = mild de-clustering.
    """

    i = Y_df.index.get_level_values(0).to_numpy()
    j = Y_df.index.get_level_values(1).to_numpy()
    dates = pd.DatetimeIndex(Y_df.index.get_level_values(2))

    t = (dates - dates.min()).total_seconds().to_numpy() / 86400
    pts = np.column_stack([
        i / radius_px, 
        j / radius_px, 
        t / time_days
    ])
    
    n = cKDTree(pts).query_ball_point(pts, r=1.0, return_length=True).astype(float)

    w = n ** -power                        
    w = np.maximum(w, w.max() / max_ratio)   
    w /= w.mean()                            
    
    return np.repeat(w[:, None], n_species, axis=1).astype(np.float32)



def _build_split(csv_data, keys, channel_dss, species_list, time_res):
    """
    csv_data    : dict {key: record}
    keys        : one of the following dicts > split["train"] / split["val"] / split["test"]
    species_list: the ~10 target species (fixes column order of Y)
    channel_dss : channel_dss : dict {channel_name: channel Dataset} (raw or normalized, only extents are used)
    time_res    : finest time resolution among the channels ("D" daily, "h" hourly, ...)

    Returns
    -------
    Y    : (N, S) float32 multi-hot labels
    W    : (N, S) float32 de-clustering weights (per sample > so each s from S has the same weight)
    meta : DataFrame with columns latitude, longitude, time (one row per sample, aligned with Y and W)
    """

    subset = {k: csv_data[k] for k in keys}

    # 1. keep only records valid for EVERY channel (same filter as the normalization stats)
    valid = set(subset)
    for ch_name, ds in channel_dss.items():
        try:
            times, _, _ = _valid_records(channel_ds=ds, csv_data=subset)
        except ValueError as e:
            raise ValueError(f"channel '{ch_name}': {e}") from e
        n_out = len(valid - set(times.index))
        if n_out:
            print(f"channel '{ch_name}': {n_out} records outside its spatial/temporal extent")
        valid &= set(times.index)
    
    if len(subset) - len(valid):
        print(f"dropping {len(subset) - len(valid)} of {len(subset)} records outside some channel's extent")

    df = pd.DataFrame.from_dict({k: subset[k] for k in valid}, orient="index")

    # 2. keep only the target species, so every sample has at least one positive
    # ALL OTHER 18 SPECIES ARE FILTERED OUT HERE
    df = df[df["scientificName"].isin(species_list)].copy()

    # 3. snap to the common grid (cell centres) and to the finest time resolution
    lat_axis = TARGET_GRID["latitude"].values
    lon_axis = TARGET_GRID["longitude"].values
    df["i"] = pd.Index(lat_axis).get_indexer(df["decimalLatitude"], method="nearest")   # idx for the nearest pixel centre (lat) in target grid, i.e. in the map
    df["j"] = pd.Index(lon_axis).get_indexer(df["decimalLongitude"], method="nearest")
    df["date"] = pd.to_datetime(df["eventDate"]).dt.floor(time_res)

    # 4. merge records of the same (cell, date) into one multi-hot row
    Y_df = (pd.crosstab([df["i"], df["j"], df["date"]], df["scientificName"])       # counts how often each species occurs for each unique (i, j, date) 
              .clip(upper=1)                                                        # if one species occurs more than once for (i, j, date) > turn into one occurrence
              .reindex(columns=species_list, fill_value=0))                         # reorders as in species_list + adds all-zero column for non-present species

    # 5. turns the crosstab’s row labels back into physical coordinates
    meta = Y_df.index.to_frame(index=False)
    meta["latitude"] = lat_axis[meta["i"].values]
    meta["longitude"] = lon_axis[meta["j"].values]
    meta = meta.rename(columns={"date": "time"})[["latitude", "longitude", "time"]]

    W = compute_weights(Y_df)
    return Y_df.values.astype(np.float32), W, meta


def build_samples(csv_data, split, channel_dss, species_list=SPECIES_LIST, time_res="D"):
    """
    fold        : one entry of data_splitter.split(csv_data), with keys "fold", "train", "val", "test"
    csv_data    : dict {key: record}
    species_list: the ~10 target species (fixes column order of Y)
    channel_dss : channel_dss : dict {channel_name: channel Dataset} (raw or normalized, only extents are used)
    time_res    : finest time resolution among the channels ("D" daily, "h" hourly, ...)


    Returns {"train": (Y, W, meta), "val": (Y, W, meta), "test": (Y, W, meta)}.
    Splits with no records (e.g. val when use_val=False) are skipped.
    """
    out = {}
    for name in ["train", "val", "test"]:
        if not split.get(name):
            continue
        out[name] = _build_split(csv_data, split[name], channel_dss, species_list, time_res)

    # leakage check: no (latitude, longitude, time) sample may appear in two splits
    seen = {n: set(map(tuple, m[["latitude", "longitude", "time"]].values)) for n, (_, _, m) in out.items()}
    names = list(seen)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            assert not (seen[names[a]] & seen[names[b]]), f"{names[a]} and {names[b]} share samples"
    return out