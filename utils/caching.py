import hashlib
from pathlib import Path


def subset_path(spec, bbox, trange, name, wider_trange=None):
    """Path to a cached subset of a dataset, unique for the spec/bbox/trange."""
    lon0, lon1, lat0, lat1 = bbox

    h = hashlib.md5((spec.location + "|".join(spec.variables or ())).encode()).hexdigest()[:6]

    parts = []
    if wider_trange is not None:
        parts.append(f"{wider_trange[0][:10]}_{wider_trange[1][:10]}")
    parts.append(f"{trange[0][:10]}_{trange[1][:10]}")
    parts.append(f"lon{lon0}_{lon1}_lat{lat0}_{lat1}")
    parts.append(h)

    dataset_id = "_".join(parts)
    cache_dir = Path("output", f"{spec.short_name}_subsets")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{name}_{dataset_id}.nc"


def cache_to_path(dir, name, suffix="pkl", tag=""):
    """Path to a cached dataset, unique for the directory and an optional tag."""
    dir = Path(dir)
    h = hashlib.md5(f"{dir}|{tag}".encode()).hexdigest()[:6]
    dir.mkdir(parents=True, exist_ok=True)
    return dir / f"{name}_{h}.{suffix}"