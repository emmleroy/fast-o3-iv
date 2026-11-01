import numpy as np
import pandas as pd
import xarray as xr

from fast_o3_iv import utils, metrics
from fast_o3_iv.config import PROCESSED_DATA_DIR

MODEL_KEY = "GPR-local"
df = utils.load_results(MODEL_KEY, "inference")

# for risk metrics (to be calculated below)
ds_risk = utils._df_to_ds(df, extra_index_cols=['ensemble_member']) 


# for standard metrics (ensemble-mean climatology and residual SD)
ds_metrics = metrics.get_spatial_pattern_of_metrics(
        df,
        metrics=[
            "ens_clim",
            "resid_sd",
        ],
    )

# reference climatology --> first 10 years of the training member
first_member = pd.read_parquet("/home/eleroy/proj-dirs/fast-o3-iv/data/processed/model_inputs/NCAR.CESM2-WACCM6.wa6_ic1.001.historical.setL.inputs.parquet")
ds_first = utils._df_to_ds(first_member)
reference_period = ds_first['mda8o3'].sel(time=slice("1975-01-01", "1984-12-31")).load()
clm = reference_period.mean("time")

# reference climatology --> first 10 years of the training member
first_member = pd.read_parquet("/home/eleroy/proj-dirs/fast-o3-iv/data/processed/model_inputs/NCAR.CESM2-WACCM6.wa6_ic1.001.historical.setL.inputs.parquet")
ds_first = utils._df_to_ds(first_member)
reference_period = ds_first['mda8o3'].sel(time=slice("1975-01-01", "1984-12-31")).load()
clm = reference_period.mean("time")

# reference grid 
ref_lat = ds_metrics['ens_clim']['val_y_pred'].lat.values
ref_lon = ds_metrics['ens_clim']['val_y_pred'].lon.values

def align_to_ref(da):
    """helper to align a data array to the reference grid, after checking
    if they're numerically the same (just float-precision-mismatched)"""
    lat_ok = da.lat.size == ref_lat.size and np.allclose(da.lat.values, ref_lat, atol=1e-4)
    lon_ok = da.lon.size == ref_lon.size and np.allclose(da.lon.values, ref_lon, atol=1e-4)
    if not (lat_ok and lon_ok):
        raise ValueError(f"grid not similar enough to the reference grid")
    return da.assign_coords(lat=ref_lat, lon=ref_lon)

# prep alignment
clm_aligned = align_to_ref(clm)
val_y_pred  = align_to_ref(ds_metrics['ens_clim']['val_y_pred'])

# predicted ensemble-mean climatology + predicted standard deviation (int var)
clim_pred = val_y_pred + clm_aligned * 1e9
sd_pred = align_to_ref(ds_metrics['resid_sd']['val_y_pred'])

# median extrapolation risk score + frequency where gpr_r > 0.75
risk_median = (ds_risk["gp_r"]).median(dim=["time", "ensemble_member"])
risk_frequency = (ds_risk["gp_r"] > 0.75).mean(dim=["time", "ensemble_member"])*100

out_ds = xr.Dataset(
    {       
        "clim_pred":    clim_pred,
        "sd_pred":      sd_pred,
        "risk_median":  risk_median,
        "risk_frequency":risk_frequency
    }
)

out_ds["clim_pred"].attrs.update(units="ppbv", description="Absolute climatology: surrogate (GPR) prediction, val_y_pred")
out_ds["sd_pred"].attrs.update(units="ppbv", description="Internal variability SD: surrogate (GPR) prediction, val_y_pred")
out_ds["risk_median"].attrs.update(units="unitless", description="Median extrapolation risk score, r (r=var_post/var_prior)")
out_ds["risk_frequency"].attrs.update(units="percent", description="Percent of samples with r>0.75")

out_path = PROCESSED_DATA_DIR / "figure_data" / "Figure5_local_metrics_fields_and_risk_scores.nc"
out_ds.to_netcdf(out_path)
print(f"Saved to {out_path}")
