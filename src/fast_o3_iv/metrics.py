# six metrics to evaluate the surrogate ensemble

## TO DO: check if we need to de-seasonalize again --> probably not?
## TO DO: check that masking is working for global_wmean and global_rmse

import xarray as xr
import numpy as np
from scipy.stats import wasserstein_distance


def calc_osdma8(ds0, targets=False):
    """
    from monthly-mda mda8 o3 in units of ppbv, calculate osdma8 
    
    key note: from Malashock et al. 2022

    OSDMA8 is calculated as the annual
    maximum of the six-month running mean of the
    monthly average daily maximum 8 h mixing ratio,
    including up to March of the following year to contain
    the Southern Hemisphere summer.
    """

    ds = ds0.copy(deep=True)
    if targets:
        rolled_vars = ['y_true']
    else:
        rolled_vars = ['y_pred'] + (['y_true'] if 'y_true' in ds.data_vars else [])

    rolled_6mo_end_labeled = ds[rolled_vars].rolling(time=6, min_periods=6).mean()
    rolled_6mo_valid = rolled_6mo_end_labeled.isel(time=slice(5, None))
    window_start_time = ds["time"].isel(time=slice(None, -5))
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

def return_internal_component(ds0):

    ds = ds0.copy(deep=True)

    resid_vars = ['y_pred'] + (['y_true'] if 'y_true' in ds.data_vars else [])
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

    # trend - note!! ds_temp is necesssary because otherwise slope is per nanosecond
    t_years = (ds_annmean['time'].dt.year
            - ds_annmean['time'].dt.year[0]) # get year index
    ds_temp = ds_annmean.assign_coords(time=t_years) # temporary data
    
    polynomial_fit = (
        ds_temp
        .polyfit(dim='time', deg=deg) 
        )
    slope = polynomial_fit.sel(degree=deg, drop=True) # slope of the polynomial fit = trend

    # set attributes
    slope.attrs['units'] = 'ppbv per year' 
    slope = slope*len(ds_annmean['time'])
    slope.attrs['units'] = f'ppbv per {str(len(ds_annmean.time))} yr'

    return slope


def calc_resid_standard_deviation(ds0):
    """local metric 3. resid. std dev (over member,time) after deseasonalizing and removing forced trend
    """
    internal = return_internal_component(ds0)
    internal_stdev = internal.std(dim=['ensemble_member', 'time']) # note! sd across members and time - not average member-deviation across time

    return internal_stdev

def calc_resid_q90(ds0, q=0.90):
    """local metric 4. resid. q90 (over member,time)
    """
    internal = return_internal_component(ds0)
    internal_q90 = internal.quantile(q, dim=['ensemble_member', 'time']) # note! q90 across member and time, not averaged q90 across time
    internal_q90 = internal_q90.reset_coords("quantile", drop=True)

    return internal_q90


def calc_prob_exceed_true_q90(ds0, q=0.90):
    """local metric 5. probability (over member,time) that internal y_pred exceeds the
    q-th percentile of internal y_true.
    """

    if 'y_true' not in ds0.data_vars:
        raise NotImplementedError("this metric is only for train/test datasets with y_true")

    internal = return_internal_component(ds0)

    # threshold: q-th percentile of TRUE (over member,time)
    true_q = internal["y_true"].quantile(q, dim=['ensemble_member', 'time'])
    true_q = true_q.reset_coords("quantile", drop=True)

    # exceedance probability for PRED relative to TRUE 
    prob_pred = (internal["y_pred"] > true_q).mean(dim=("ensemble_member", "time"))

    # sanity check (should be ~0.1 if enough samples)
    prob_true = (internal["y_true"] > true_q).mean(dim=("ensemble_member", "time"))

    prob_q90 = xr.Dataset(
        {
            "true_q90": true_q,
            "prob_pred_exceed_true_q90": prob_pred,
            "prob_true_exceed_true_q90": prob_true,
        }
    )

    return prob_q90


def calc_emd_to_noise(ds0):
    """local metric 6. calculate the emd to noise ratio of val_y_pred relative to val_y_true
    """
    if 'y_true' not in ds0.data_vars:
        raise NotImplementedError("this metric is only for train/test datasets with y_true")

    internal = return_internal_component(ds0)
    dims=('ensemble_member', 'time')

    def _emd(a, b):
        a = np.asarray(a).ravel()
        b = np.asarray(b).ravel()
        return wasserstein_distance(a, b)
   
    # vectorized EMD across all remaining dimensions
    emd = xr.apply_ufunc(
        _emd,
        internal['y_pred'],
        internal['y_true'],
        input_core_dims=[list(dims), list(dims)],
        output_dtypes=[float],
        vectorize=True,
        dask="parallelized",
    )
    emd.name = "emd"
    emd.attrs.update({"metric": "1-Wasserstein", "reduced_over": ",".join(dims)})

    true_internal_std = internal['y_true'].std(dim=list(dims))

    emd_to_noise = xr.Dataset(
        {
            "emd_to_noise": emd/true_internal_std,
        }
    )

    return emd_to_noise
