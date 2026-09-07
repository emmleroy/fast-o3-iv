import numpy as np
import pandas as pd
import xarray as xr

from fast_o3_iv import utils, metrics
from fast_o3_iv.config import PROCESSED_DATA_DIR


MODEL_KEYS = [
    "GPR-local",
]

for model_key in MODEL_KEYS:
    df = utils.load_results(model_key, "testing")
    ds = metrics.get_spatial_pattern_of_metrics(df)


# reference climatology --> first 10 years of the training member
first_member = pd.read_parquet("/home/eleroy/proj-dirs/fast-o3-iv/data/processed/model_inputs/NCAR.CESM2-WACCM6.wa6_ic1.001.historical.setL.inputs.parquet")
ds_first = utils._df_to_ds(first_member)
reference_period = ds_first['mda8o3'].sel(time=slice("1975-01-01", "1984-12-31")).load()
clm = reference_period.mean("time")

# reference grid 
ref_lat = ds['ens_clim']['val_y_true'].lat.values
ref_lon = ds['ens_clim']['val_y_true'].lon.values

def align_to_ref(da):
    """helper to align a data array to the reference grid, after checking
    if they're numerically the same (just float-precision-mismatched)"""
    lat_ok = da.lat.size == ref_lat.size and np.allclose(da.lat.values, ref_lat, atol=1e-4)
    lon_ok = da.lon.size == ref_lon.size and np.allclose(da.lon.values, ref_lon, atol=1e-4)
    if not (lat_ok and lon_ok):
        raise ValueError(f"grid not similar enough to the reference grid")
    return da.assign_coords(lat=ref_lat, lon=ref_lon)

# trend fields
trend_true = align_to_ref(ds['ens_trend']['val_y_true_polyfit_coefficients'])
trend_pred = align_to_ref(ds['ens_trend']['val_y_pred_polyfit_coefficients'])
trend_diff = trend_pred - trend_true

# climatology fields
clm_aligned = align_to_ref(clm, "clm")
val_y_true  = align_to_ref(ds['ens_clim']['val_y_true'], "val_y_true (ens_clim)")
val_y_pred  = align_to_ref(ds['ens_clim']['val_y_pred'], "val_y_pred (ens_clim)")

# add reference climatology back
clim_true = val_y_true + clm_aligned * 1e9
clim_pred = val_y_pred + clm_aligned * 1e9
clim_diff = clim_pred - clim_true

out_ds = xr.Dataset(
    {
        "clim_true":  clim_true,
        "clim_pred":  clim_pred,
        "clim_diff":  clim_diff,
        "trend_true": trend_true,
        "trend_pred": trend_pred,
        "trend_diff": trend_diff,
    }
)

# attach metadata
out_ds["clim_true"].attrs.update(units="ppbv", description="Absolute climatology: CESM2-WACCM6 ensemble mean, val_y_true")
out_ds["clim_pred"].attrs.update(units="ppbv", description="Absolute climatology: surrogate (GPR) prediction, val_y_pred")
out_ds["clim_diff"].attrs.update(units="ppbv", description="clim_pred - clim_true")
out_ds["trend_true"].attrs.update(units="ppbv per 40yr", description="CESM2-WACCM6 ensemble-mean linear trend, val_y_true")
out_ds["trend_pred"].attrs.update(units="ppbv per 40yr", description="Surrogate (GPR) linear trend, val_y_pred")
out_ds["trend_diff"].attrs.update(units="ppbv per 40yr", description="trend_pred - trend_true")

out_path = PROCESSED_DATA_DIR / "figure_data" / "Figure2_local_metrics_fields.nc"
out_ds.to_netcdf(out_path)
print(f"Saved to {out_path}")
