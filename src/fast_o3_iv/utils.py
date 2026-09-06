import os

import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import rioxarray as rxr
import xarray as xr
#import xesmf as xe

from fast_o3_iv.config import PROCESSED_DATA_DIR
from fast_o3_iv.config import INTERIM_DATA_DIR, EXTERNAL_DATA_DIR
from fast_o3_iv.preprocess.model_configs import MODEL_CONFIG


POP_THRESH = 1.0 # 1 person/km2
SPATIAL_DIMS = ("lat", "lon")

def set_matplotlib_font(font_family):
    """Set the matplotlib font family."""
    #assert font_family in supported_fonts, f'Font {font_family} not supported.'
    plt.rcParams['font.family'] = font_family
    plt.rcParams.update({'mathtext.default': 'regular'})

def _df_to_ds(df, extra_index_cols=None):
    """Convert a pandas dataframe to an xarray dataset

    Also assumes the dataframe has a "year" and "month" column
    and converts that into a "time" column.

    Args:
        df: dataframe with lat, lon, year, month columns (plus data columns).
        extra_index_cols: optional list of additional columns to include in
            the index alongside lat, lon, time (e.g. ["ensemble_member"]).
    """
    df = df.copy()
    df["time"] = pd.to_datetime(
        df["year"] * 10000 + df["month"] * 100 + 1,
        format="%Y%m%d",
        cache=True,
    )

    index_cols = ["lat", "lon", "time"] + (extra_index_cols or [])

    ds = (
        df.drop(columns=["year", "month"])
        .set_index(index_cols)
        .to_xarray()
    )

    return ds

def add_grid_weights(ds, grid):
    """Add grid weights to an xr.Dataset.

    Currently implemented only for CESM2-WACCM6_0.9x1.25.
    also works when ds is a latitude subset of that grid.
    """

    if grid != "CESM2-WACCM6_0.9x1.25":
        raise NotImplementedError(
            "sorry!! only CESM2-WACCM6_0.9x1.25 grid accepted"
        )

    gw_path = INTERIM_DATA_DIR / "NCAR/inputdata/gw/gw_CESM2-WACCM6_0.9x1.25.nc"
    gw_ds = xr.open_dataset(gw_path, engine="netcdf4")
    gw = gw_ds["gw"]

    lat_ds = ds["lat"]
    lat_gw = gw["lat"]

    # Case 1: exact match
    if lat_ds.equals(lat_gw):
        return ds.assign(gw=gw)

    # Case 2: ds is a subset / reordered version of the same grid,
    # but coords may differ by tiny float noise
    try:
        gw_subset = gw.sel(lat=lat_ds, method="nearest", tolerance=1e-6)
    except KeyError:
        raise ValueError(
            "Latitude coordinates in ds could not be matched to gw grid "
            "within tolerance."
        )

    # reattach ds lat coordinates exactly, so downstream equality checks work
    gw_subset = gw_subset.assign_coords(lat=lat_ds)

    # final safety check
    if gw_subset.sizes["lat"] != lat_ds.size:
        raise ValueError("Failed to align grid weights to ds lat dimension.")

    return ds.assign(gw=gw_subset)

def add_population_density(ds):
    """Add population density as a variable to an xr.Dataset
    """
    pop_path = INTERIM_DATA_DIR / "GPWv4/gpw_v4_population_density_rev11_2015_192_288.nc"
    pop = xr.open_dataset(pop_path)["population"].rename("population_density")

    pop = pop.sel(lat=ds.lat, lon=ds.lon, method="nearest", tolerance=1e-4)
    pop = pop.assign_coords(lat=ds.lat, lon=ds.lon)

    ds = ds.copy()
    ds["population_density"] = pop
    ds["population_density"].attrs.update({
        "long_name": "Population density",
        "standard_name": "population_density",
        "units": "persons km-2",
    })
    return ds

def add_aux_data(ds):
    """Add auxilary data (grid weights and population density) data to netcdf results"""
    ds = add_grid_weights(ds, grid="CESM2-WACCM6_0.9x1.25")
    ds = add_population_density(ds)
    return ds

def pop_mask(ds, thresh=POP_THRESH):
    """Define a boolean mask where population density exceeds threshold"""
    return ds["population_density"] > thresh

def drop_unpopulated_cells(df):
    """Drop unpopulated cells from a pandas DataFrame."""
    ds = _df_to_ds(df)
    ds = add_aux_data(ds)

    mask = pop_mask(ds)
    ds_masked = ds.where(mask)

    df_masked = ds_masked.to_dataframe().dropna()
    return df_masked

def safe_weights(ds, like, mask):
    """
    return a weights dataset where values are zero where a given mask is False
    and where values are zero where the data are missing
    """

    # 'gw' is lat-only so we broadcast to a similar array
    w = ds["gw"].broadcast_like(like)

    if mask is not None:

        # set weights outside the mask to zero, if a mask exists
        w = w.where(mask, 0)

    # also exclude where the data are missing
    w = w.where(like.notnull(), 0)

    # fill any other nans in the gws to zero
    return w.fillna(0)

def area_weighted_mean(ds, var, spatial_dims=SPATIAL_DIMS, mask=None):
    """calculate the area weighted mean of variable 'var' in dataset 'ds' 
       with or without a mask"""
    da = ds[var]
    w = safe_weights(ds, da, mask=mask)
    return da.weighted(w).mean(spatial_dims)


###############################################



#def has_expected_data(path, expected_timesteps):
#    try:
#        with xr.open_dataset(path, engine="netcdf4", decode_cf=False) as ds:
#            if ds.sizes.get("time") != expected_timesteps:
#                return False
#            if len(ds.data_vars) == 0:
#                return False
#            return True
#    except Exception:
#        return False

#def exists(path):
#    return os.path.exists(path) and os.path.getsize(path) > 0




#def get_mda8o3_climatology(
#        config,
#    ):
    
#    path = (
#        INTERIM_DATA_DIR 
#        / config.institution_id
#        / config.source_id
#        / config.cmip_experiment_id
#        / config.first_member
#        / 'mda8o3'
#    )
#    file_name = (
#            f"{config.institution_id}."
#            f"{config.source_id}."
#            f"{config.cmip_experiment_id}."
#            f"{config.first_member}."
#            "mda8o3."
#            f"{config.year_range[0]}-{config.year_range[-1]}.nc"
#        )
#    ds = xr.open_dataset(path / file_name)
#    ref_period = (ds['mda8o3']*1e9).sel(
#            time=slice(f"{str(config.year_range[0])}-01-01", 
#            f"{str(config.year_range[0]+9)}-12-31")).load()#

#    ref_monthly_clm = ref_period.groupby("time.month").mean("time")
#    ref_monthly_clm.attrs["units"] = "ppbv"
#    return ref_monthly_clm

#def add_mda8o3_climatology(ds0, train_config):
#    clm = get_mda8o3_climatology(train_config)

#    assert clm.sizes["lat"] == ds0.sizes["lat"]
#    assert clm.sizes["lon"] == ds0.sizes["lon"]

#    clm2 = clm.assign_coords(lat=ds0.lat, lon=ds0.lon)
#    ds0_clm = ds0.groupby("time.month") + clm2

#    return ds0_clm

