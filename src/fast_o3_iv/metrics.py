import xarray as xr
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from fast_o3_iv import utils

# -----------------------------------------------------------------------
# Validation metrics: RMSE, MAE, Bias

_LOCAL_ERROR_FUNCS = {
    "rmse": lambda pred, true: (pred - true) ** 2, # mse, square root taken after mean
    "mae": lambda pred, true: np.abs(pred - true),
    "bias": lambda pred, true: pred - true,
}

def _local_metric_time(ds, metric, pred_var, true_var, time_dim="time"):
    """Compute a local (per-grid-cell) error metric, averaged over time.
    """

    error_da = _LOCAL_ERROR_FUNCS[metric](ds[pred_var], ds[true_var])
    metric_da = error_da.mean(dim=time_dim, skipna=True)

    if metric == "rmse":
        metric_da = np.sqrt(metric_da)

    metric_ds = metric_da.to_dataset(name=metric)
    return metric_ds.merge(ds[["gw", "population_density"]])


def _area_weighted_mean_of_local_metric(ds, metric, mask=None):
    """Compute the global, area-weighted mean of a local (per-grid-cell) error metric.
    """
    area_wmean = utils.area_weighted_mean(ds, metric, mask)
    area_wmean_ds = area_wmean.to_dataset(name=metric)
    area_wmean_ds.attrs.update({"metric": f"global area-weighted mean of local {metric}"})
    return area_wmean_ds[metric].values


def calculate_global_wmean_of_local_metric(df, metric, only_populated_grid_cells=True):
    """Compute the global area-weighted mean metric directly from a dataframe.

    Converts df to an xarray Dataset, attaches grid weights
    and population density, computes the local (per-grid-cell) metric over time from
    val_y_pred/val_y_true, then reduces it to a single global area-weighted value.

    Args:
        df: dataframe with lat, lon, year, month, val_y_pred, and
            val_y_true columns.
        metric: One of "rmse", "mae", "bias" (see local_metric_time).
        only_populated_grid_cells: If True, restrict the mean to grid cells above
            utils.POP_THRESH population density.

    Returns:
        The scalar global area-weighted mean metric as a numpy array.
    """

    # convert dataframe to a dataset
    ds = utils._df_to_ds(df)

    # add auxilary data (population density and grid weights)
    ds = utils.add_aux_data(ds)

    # grab the population mask if only_populated_grid_cells=True
    mask = utils.pop_mask(ds) if only_populated_grid_cells else None

    # calculate the local metric per grid cell
    local_metric_ds = _local_metric_time(ds, metric, "val_y_pred", "val_y_true")

    # return the area-weighted mean of the local metric
    return _area_weighted_mean_of_local_metric(local_metric_ds, metric=metric, mask=mask)

# -----------------------------------------------------------------------
# Test metrics: 
# ensemble mean climatology
# ensemble mean linear trend
# internal variablity SD
# internal variability q90
# internal variability P(>q90)
# internal variability EMD-to-noise

METRIC_SPECS = {
    "ens_clim": ("rmse", "val_y_pred", "val_y_true"),
    "ens_trend": ("rmse", "val_y_pred_polyfit_coefficients", "val_y_true_polyfit_coefficients"),
    "resid_sd": ("rmse", "val_y_pred", "val_y_true"),
    "resid_q90": ("rmse", "val_y_pred", "val_y_true"),
    "pgt_q90": ("rmse", "prob_pred_exceed_true_q90", "prob_true_exceed_true_q90"),
    "emd_to_noise": ("wmean", "emd_to_noise", None),
}


def return_internal_component(ds0):

    ds = ds0.copy(deep=True)

    resid_vars = ['val_y_pred'] + (['val_y_true'] if 'val_y_true' in ds.data_vars else [])
    ds = ds[resid_vars]

    if len(ds.time)==480: # de-seasonalize monthly-mean mda8 o3
        ds_deseasonalized = ds.groupby('time.month') - ds.groupby('time.month').mean()
    elif len(ds.time)==40:
        ds_deseasonalized = ds #osdma8 is already annualized (no need to de-seasonalize)
    else:
        raise NotImplementedError

    forced = ds_deseasonalized.mean(dim='ensemble_member')
    internal = ds_deseasonalized-forced

    return internal


def calc_ensemble_mean_clim(ds):
    """local metric 1: ensemble mean climatology"""
    return ds.mean(dim=["ensemble_member",'time'])


def calc_ensemble_mean_trend(ds, deg=1):
    """local metric 2: calculate the linear trend of annual means in each grid cell
    using least squares polynomial fit.
    """
    if len(ds.time)==480:
        ds_annmean = ds.mean(dim='ensemble_member').resample(time='1Y').mean('time') # annual mean
    elif len(ds.time)==40:
        ds_annmean = ds.mean(dim='ensemble_member') # already annualized
    else:
        raise NotImplementedError

    t_years = (ds_annmean['time'].dt.year - ds_annmean['time'].dt.year[0])
    ds_tmp = ds_annmean.assign_coords(time=t_years)

    polynomial_fit = ds_tmp.polyfit(dim='time', deg=deg)
    slope = polynomial_fit.sel(degree=deg, drop=True)

    slope.attrs['units'] = 'ppbv per year'
    slope = slope*len(ds_annmean['time'])
    slope.attrs['units'] = f'ppbv per {str(len(ds_annmean.time))} yr'

    return slope


def calc_resid_standard_deviation(ds0):
    """local metric 3. resid. std dev (over member,time) after deseasonalizing and removing forced trend"""
    internal = return_internal_component(ds0)
    return internal.std(dim=['ensemble_member', 'time'])


def calc_resid_q90(ds0, q=0.90):
    """local metric 4. resid. q90 (over member,time)"""
    internal = return_internal_component(ds0)
    internal_q90 = internal.quantile(q, dim=['ensemble_member', 'time'])
    return internal_q90.reset_coords("quantile", drop=True)


def calc_prob_exceed_true_q90(ds0, q=0.90):
    """local metric 5. probability (over member,time) that internal val_y_pred exceeds the
    q-th percentile of internal val_y_true.
    """
    if 'val_y_true' not in ds0.data_vars:
        raise NotImplementedError("this metric is only for train/test datasets with y_true")

    internal = return_internal_component(ds0)

    true_q = internal["val_y_true"].quantile(q, dim=['ensemble_member', 'time'])
    true_q = true_q.reset_coords("quantile", drop=True)

    prob_pred = (internal["val_y_pred"] > true_q).mean(dim=("ensemble_member", "time"))
    prob_true = (internal["val_y_true"] > true_q).mean(dim=("ensemble_member", "time"))  # sanity check, should be ~0.1

    return xr.Dataset({
        "true_q90": true_q,
        "prob_pred_exceed_true_q90": prob_pred,
        "prob_true_exceed_true_q90": prob_true,
    })


def calc_emd_to_noise(ds0):
    """local metric 6. calculate the emd to noise ratio of val_y_pred relative to val_y_true"""
    if 'val_y_true' not in ds0.data_vars:
        raise NotImplementedError("this metric is only for train/test datasets with y_true")

    internal = return_internal_component(ds0)
    dims = ('ensemble_member', 'time')

    def _emd(a, b):
        return wasserstein_distance(np.asarray(a).ravel(), np.asarray(b).ravel())

    emd = xr.apply_ufunc(
        _emd,
        internal['val_y_pred'],
        internal['val_y_true'],
        input_core_dims=[list(dims), list(dims)],
        output_dtypes=[float],
        vectorize=True,
        dask="parallelized",
    )
    emd.name = "emd"
    emd.attrs.update({"metric": "1-Wasserstein", "reduced_over": ",".join(dims)})

    true_internal_std = internal['val_y_true'].std(dim=list(dims))

    return xr.Dataset({"emd_to_noise": emd / true_internal_std})


def compute_local_metric_fields(
    base_ds,
    metrics=[
        "ens_clim",
        "ens_trend",
        "resid_sd",
        "resid_q90",
        "pgt_q90",
        "emd_to_noise",
    ],
):
    """return xr.Datasets for the requested local metrics,
    with gw + population_density merged back on for downstream masking.
    """
    aux = base_ds[["gw", "population_density"]]

    metric_funcs = {
        "ens_clim": calc_ensemble_mean_clim,
        "ens_trend": calc_ensemble_mean_trend,
        "resid_sd": calc_resid_standard_deviation,
        "resid_q90": calc_resid_q90,
        "pgt_q90": calc_prob_exceed_true_q90,
        "emd_to_noise": calc_emd_to_noise,
    }

    return {
        name: metric_funcs[name](base_ds).merge(aux)
        for name in metrics
    }


def global_rmse(ds, pred_var, true_var, mask=None):
    """calculate the global area-weighted rmse between pred and true statistics"""
    pred, true = ds[pred_var], ds[true_var]
    w = utils.safe_weights(ds, pred, mask=mask)
    mse = ((pred - true) ** 2).weighted(w).mean(utils.SPATIAL_DIMS)
    return np.sqrt(mse)


def summarize_global_metrics(metric_dsets, only_populated_grid_cells=True):
    """take as input dictionary of local metrics and return summary of global metrics (global rmse or global wmean)"""
    metrics_summary = {}
    for name, ds_metric in metric_dsets.items():
        kind, a, b = METRIC_SPECS[name]
        mask = utils.pop_mask(ds_metric) if only_populated_grid_cells else None

        if kind == "rmse":
            metrics_summary[name] = global_rmse(ds_metric, a, b, mask=mask)
        elif kind == "wmean":
            metrics_summary[name] = utils.area_weighted_mean(ds_metric, a, mask=mask)
        else:
            raise ValueError(f"Unknown summary kind: {kind}")

    return metrics_summary


def get_spatial_pattern_of_metrics(
        df,
        metrics=[
            "ens_clim",
            "ens_trend",
            "resid_sd",
            "resid_q90",
            "pgt_q90",
            "emd_to_noise",
        ],
    ):
    """Return a spatial pattern of the test metrics.

    Convert df to xarray Dataset, attach grid weights and population density, compute each local
    (per-grid-cell) test metric.

    """
    ds = utils._df_to_ds(df, extra_index_cols=["ensemble_member"])
    ds = utils.add_aux_data(ds)

    local_metric_dsets = compute_local_metric_fields(ds, metrics)
    
    return local_metric_dsets


def calculate_global_test_metrics(df, only_populated_grid_cells=True):
    """Compute all global area-weighted test metrics directly from a dataframe.

    Convert df to xarray Dataset, attach grid weights and population density, compute each local
    (per-grid-cell) test metric, then reduce each to a
    single global area-weighted value.

    """

    local_metric_dsets = get_spatial_pattern_of_metrics(df)
    return summarize_global_metrics(local_metric_dsets, only_populated_grid_cells=only_populated_grid_cells)


def calc_osdma8(da):
    """
    from monthly-mda mda8 o3 in units of ppbv, calculate osdma8 
    
    key note: from Malashock et al. 2022

    OSDMA8 is calculated as the annual
    maximum of the six-month running mean of the
    monthly average daily maximum 8 h mixing ratio,
    including up to March of the following year to contain
    the Southern Hemisphere summer.
    """

    da = da.copy(deep=True)

    rolled_6mo_end_labeled = da.rolling(time=6, min_periods=6).mean()
    rolled_6mo_valid = rolled_6mo_end_labeled.isel(time=slice(5, None))
    window_start_time = da["time"].isel(time=slice(None, -5))
    rolled_6mo_start_labeled = rolled_6mo_valid.assign_coords(time=window_start_time)

    # keep only windows starting Jan..Oct
    rolled_jan_oct = rolled_6mo_start_labeled.where(
        rolled_6mo_start_labeled.time.dt.month <= 10, drop=True
    )

    # annual max by START year (and avoid collision with existing 'year' var!)
    osdma8 = rolled_jan_oct.resample(time='YS').max()

    osdma8.attrs.update({
            "long_name" : "Ozone season daily 8-hour mean ozone (OSDMA8)",
            "units" : "ppbv",
        })
    return osdma8
