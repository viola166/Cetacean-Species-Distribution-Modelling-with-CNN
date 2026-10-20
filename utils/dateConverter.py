import pandas as pd

def date_to_index(ds, date):
    """Position in ds's time axis of the timestamp closest to 'date'."""
    return ds.get_index("time").get_indexer([pd.Timestamp(date)], method="nearest")[0]