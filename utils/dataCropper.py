import pandas as pd
import time
import xarray as xr
import numpy as np
import requests

def widen_bbox(bbox, wide):
    """Smallest box containing both `bbox` and `wide` (lon0, lon1, lat0, lat1)."""
    if wide is None:
        return bbox
    return (min(bbox[0], wide[0]), max(bbox[1], wide[1]),
            min(bbox[2], wide[2]), max(bbox[3], wide[3]))


def widen_trange(trange, wide, time_offset):
    """Smallest range containing `trange` and `wide`, padded back by the window."""
    t0 = pd.Timestamp(trange[0])
    t1 = pd.Timestamp(trange[1])

    if wide is not None:
        t0 = min(t0, pd.Timestamp(wide[0]))
        t1 = max(t1, pd.Timestamp(wide[1]))

    if time_offset is not None:
        t0 = t0 - pd.Timedelta(days=int(time_offset) - 1)

    return (t0.isoformat(), t1.isoformat())


def nearest_date(ds, trange, in_chunks=False):
    start_dt, end_dt = trange
    start, end = (pd.Timestamp(t).normalize() for t in trange)
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()

    if start == end: #and start >= today:          # forecast request: nearest available step
        subset = ds.sel(time=[pd.Timestamp(start_dt)], method="nearest")
        gap = abs(pd.Timestamp(subset.time.values[0]) - pd.Timestamp(start_dt))
        print(f"Nearest step is {gap} from {start_dt}")
        if gap > pd.Timedelta(days=5):           # don't silently use stale data
            # raise ValueError(
            #     f"Nearest step is {gap} from {start_dt}; "
            #     f"data ends {ds.time.max().values}")
            subset = average_day(ds, start_dt)
    elif in_chunks is False:
        subset = ds.sel(time=slice(start_dt, end_dt))
    elif in_chunks is True:
        subset = load_in_time_chunks(ds, trange)
    return subset

def average_day(ds, day, doy_window=0, step=1):     # With pydap, xarray downloads everything between the first and last index and picks the rows afterwards
    target = pd.Timestamp(day)
    doy = ds["time"].dt.dayofyear
    diff = np.abs((doy - target.dayofyear + 183) % 366 - 183)
    idx = np.flatnonzero(diff.values <= doy_window)    # ndarray of positions

    if len(idx) == 0:                                   # not idx.sizes[...]
        raise ValueError(
            f"No data for day-of-year {target.dayofyear} "
            f"(+/- {doy_window}); data ends {ds.time.max().values}")

    print(f"Using day-of-year mean ({len(idx)} samples) instead of the nearest step")

    total, count = None, None
    for i in range(0, len(idx), step):
        print(f"Loading {i}-{min(i + step, len(idx))} of {len(idx)}")
        part = robust_load(ds.isel(time=idx[i:i + step]))
        s = part.sum("time", skipna=True)
        c = part.notnull().sum("time")
        total = s if total is None else total + s
        count = c if count is None else count + c

    return (total / count.where(count > 0)).expand_dims(time=[target])


RETRYABLE = (OSError, TimeoutError, ConnectionError, requests.exceptions.RequestException)

def robust_load(ds, retries=5, wait=10):
    for i in range(retries):
        try:
            return ds.load()
        except RETRYABLE as e:  # requests.ConnectionError / TimeoutError
            if i == retries - 1:
                raise
            print(f"Attempt {i+1} failed ({e.__class__.__name__}), retrying in {wait*(i+1)}s")
            time.sleep(wait * (i + 1))


def load_in_time_chunks(ds, trange, step=30):
    print("in load in time chunks")
    ds = ds.sel(time=slice(*trange))          # label-based: dates, inclusive on both ends
    n = ds.sizes["time"]
    if n == 0:
        raise ValueError(f"No data in time range {trange}")

    parts = []
    for i in range(0, n, step):               # integer positions from here on
        print(f"Loading time steps {i}-{min(i + step, n)} of {n}")
        parts.append(robust_load(ds.isel(time=slice(i, i + step))))
    return xr.concat(parts, dim="time")