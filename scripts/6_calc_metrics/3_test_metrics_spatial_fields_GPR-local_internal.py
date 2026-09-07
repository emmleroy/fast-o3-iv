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

# reference grid (note not the same as fig2)
ref_lat = ds['resid_sd']['val_y_true'].lat.values
ref_lon = ds['resid_sd']['val_y_true'].lon.values

def align_to_ref(da):
    """helper to align a data array to the reference grid, after checking
    if they're numerically the same (just float-precision-mismatched)"""
    lat_ok = da.lat.size == ref_lat.size and np.allclose(da.lat.values, ref_lat, atol=1e-4)
    lon_ok = da.lon.size == ref_lon.size and np.allclose(da.lon.values, ref_lon, atol=1e-4)
    if not (lat_ok and lon_ok):
        raise ValueError(f"grid not similar enough to the reference grid")
    return da.assign_coords(lat=ref_lat, lon=ref_lon)

# internal variability: SD
sd_true = align_to_ref(ds['resid_sd']['val_y_true'])
sd_pred = align_to_ref(ds['resid_sd']['val_y_pred'])
sd_diff = sd_pred - sd_true

# internal variability: q90 (anomaly -> absolute)
clm_aligned = align_to_ref(clm, "clm")
q90_true = align_to_ref(ds['resid_q90']['val_y_true'], "q90_true") + clm_aligned * 1e9
q90_pred = align_to_ref(ds['resid_q90']['val_y_pred'], "q90_pred") + clm_aligned * 1e9
q90_diff = q90_pred - q90_true

# internal variability: P(>q90)
pgtq90_true = align_to_ref(ds['pgt_q90']['prob_true_exceed_true_q90'], "pgtq90_true")
pgtq90_pred = align_to_ref(ds['pgt_q90']['prob_pred_exceed_true_q90'], "pgtq90_pred")
pgtq90_diff = pgtq90_pred - pgtq90_true

# internal variability: EMD-to-noise (single field)
emdtonoise = align_to_ref(ds['emd_to_noise']['emd_to_noise'], "emdtonoise")

out_ds = xr.Dataset(
    {
        "sd_true":      sd_true,
        "sd_pred":      sd_pred,
        "sd_diff":      sd_diff,
        "q90_true":     q90_true.drop_vars(["month"], errors="ignore"),
        "q90_pred":     q90_pred.drop_vars(["month"], errors="ignore"),
        "q90_diff":     q90_diff.drop_vars(["month"], errors="ignore"),
        "pgtq90_true":  pgtq90_true,
        "pgtq90_pred":  pgtq90_pred,
        "pgtq90_diff":  pgtq90_diff,
        "emdtonoise":   emdtonoise,
    }
)

# attach metadata
out_ds["sd_true"].attrs.update(units="ppbv", description="Internal variability SD: CESM2-WACCM6, val_y_true")
out_ds["sd_pred"].attrs.update(units="ppbv", description="Internal variability SD: surrogate (GPR) prediction, val_y_pred")
out_ds["sd_diff"].attrs.update(units="ppbv", description="sd_pred - sd_true")
out_ds["q90_true"].attrs.update(units="ppbv", description="Internal variability q90 (absolute): CESM2-WACCM6, val_y_true")
out_ds["q90_pred"].attrs.update(units="ppbv", description="Internal variability q90 (absolute): surrogate (GPR) prediction, val_y_pred")
out_ds["q90_diff"].attrs.update(units="ppbv", description="q90_pred - q90_true")
out_ds["pgtq90_true"].attrs.update(units="unitless", description="P(true > true q90)")
out_ds["pgtq90_pred"].attrs.update(units="unitless", description="P(pred > true q90)")
out_ds["pgtq90_diff"].attrs.update(units="unitless", description="pgtq90_pred - pgtq90_true")
out_ds["emdtonoise"].attrs.update(units="unitless", description="Earth mover's distance to noise")


out_path = PROCESSED_DATA_DIR / "figure_data" / "Figure3_local_metrics_fields.nc"
out_ds.to_netcdf(out_path)
print(f"Saved to {out_path}")
