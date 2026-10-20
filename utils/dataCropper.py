import pandas as pd

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