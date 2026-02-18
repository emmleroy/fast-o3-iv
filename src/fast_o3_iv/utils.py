import os
import xarray as xr

def has_expected_data(path, expected_timesteps):
    try:
        with xr.open_dataset(path, engine="netcdf4", decode_cf=False) as ds:
            if ds.sizes.get("time") != expected_timesteps:
                return False

            if len(ds.data_vars) == 0:
                return False

            return True
    except Exception:
        return False

def exists(path):
    return os.path.exists(path) and os.path.getsize(path) > 0
