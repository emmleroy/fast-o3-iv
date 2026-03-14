import os

import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe
import rioxarray as rxr

from fast_o3_iv.config import PROCESSED_DATA_DIR
from fast_o3_iv.config import INTERIM_DATA_DIR, EXTERNAL_DATA_DIR
from fast_o3_iv.preprocess.model_configs import MODEL_CONFIG


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

def _aggregated_results_ds_dir(
    mode, config, model_key
):
    return (
        PROCESSED_DATA_DIR
        / "model_outputs"
        / "aggregated_results"
        / config.institution_id
        / config.source_id
        / config.cmip_experiment_id
        / mode
        / config.variable_setname
        / model_key
    )

def _aggregated_results_path(
    mode, config, model_key
):
    out_dir = _aggregated_results_ds_dir(
        mode, config, model_key
    )

    file_name = (
        f"{config.institution_id}."
        f"{config.source_id}."
        f"{config.cmip_experiment_id}."
        f"{mode}."
        f"{config.variable_setname}."
        f"{model_key}.nc"
    )
    return out_dir / file_name


def _parquet_file_to_ds(path):
    ensemble_member = path.parts[-4]
    print(ensemble_member, flush=True)

    df = pd.read_parquet(
        path,
        columns=["lat", "lon", "year", "month", "y_pred", "y_true"],
    ).copy()

    ds = _df_to_ds(df)

    ds = ds.expand_dims(ensemble_member=[ensemble_member])

    return ds

def _df_to_ds(df):
    df = df.copy()
    df["time"] = pd.to_datetime(
        df["year"] * 10000 + df["month"] * 100 + 1,
        format="%Y%m%d",
        cache=True,
    )

    ds = (
        df.drop(columns=["year", "month"])
        .set_index(["lat", "lon", "time"])
        .to_xarray()
    )

    return ds

def _get_member_result_files(
    mode,              # validation, testing, inference
    config,
    model_key,         # GPR-local, MLR-local
):

    if mode == "validation":
        member_list = config.second_member
    elif mode == "testing":
        member_list = config.remaining_members
    elif mode == "inference":
        member_list = config.members # all members
    else:
        raise NotImplementedError("mode must be one of ['validation', 'testing', 'inference']")

    if isinstance(member_list, str):
        member_list = [member_list] # iterable

    files = sorted(
        p
        for member in member_list
        for p in PROCESSED_DATA_DIR.glob(
            (
                f"model_outputs/results/"
                f"{config.institution_id}/"
                f"{config.source_id}/"
                f"{config.cmip_experiment_id}/"
                f"{member}/"
                f"{config.variable_setname}/"
                f"{model_key}/*"
            )
        )
        if p.is_file()
    )

    print(f"Found {len(files)} parquet files")
    for f in files:
        print(f)

    return files

def aggregate_results(
    mode,              # validation, testing, inference
    config,
    model_key,         # GPR-local, MLR-local
):
    files = _get_member_result_files(mode, config, model_key)
    datasets = [_parquet_file_to_ds(f) for f in files]

    ds = xr.concat(
        datasets,
        dim="ensemble_member",
        data_vars="minimal",
        coords="minimal",
        compat="override",
        join="exact",
        combine_attrs="override",
    )

    out_path = _results_ds_path(
        mode, config, model_key
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving to {out_path}")
    ds.to_netcdf(out_path)

    return ds

def load_or_aggregate_results(
    mode, config, model_key
):
    out_path = _aggregated_results_path(
        mode, config, model_key
    )

    if exists(out_path):
        print(f"Loading existing dataset from {out_path}")
        return xr.open_dataset(out_path)

    print("No existing dataset found. Creating it now.")
    return aggregate_results(
        mode, config, model_key
    )

def add_grid_weights(ds, grid):
    """Add grid weights to an xr.Dataset
    only implemented for CESM2-WACCM6_0.9x1.25 grid
    """

    if grid != 'CESM2-WACCM6_0.9x1.25':
        raise NotImplementedError("sorry! only CESM2-WACCM6_0.9x1.25 grid accepted")

    # open raw gridweight data (saved in ExtData)
    gw_path = INTERIM_DATA_DIR / "NCAR/inputdata/gw/gw_CESM2-WACCM6_0.9x1.25.nc"
    gw_ds = xr.open_dataset(gw_path, engine='netcdf4')
    gw = gw_ds["gw"]

    lat_ds = ds["lat"]
    lat_gw = gw["lat"]

    # fast path: exact coordinate match
    if lat_ds.equals(lat_gw):
        return ds.assign(gw=gw)

    # safe index-align if grids are effectively identical
    if (lat_ds.size == lat_gw.size) and np.allclose(lat_ds.values, lat_gw.values, rtol=0, atol=1e-7):
        gw = gw.assign_coords(lat=lat_ds)
        return ds.assign(gw=gw)

    # raise error if real mismatch
    raise ValueError("Latitude coordinates do not match! refusing to assign gw.")

    return ds

#def add_population_density(ds):
    #"""Add population density to an xr.Dataset from GPWv4 (gpw_v4_population_density_rev11_2015_1_deg)

    #to do: fix bug (kernel crashes when building regridder object)
    #"""

    # load raw GPWv4 tiff file
    #gpw_path = EXTERNAL_DATA_DIR / "GPWv4/gpw_v4_population_density_rev11_2015_1_deg.tif"
    #population_tiff = rxr.open_rasterio(gpw_path)

    # sqeeze band and rename x/y to lon/lat
    #population_tiff = population_tiff.squeeze("band", drop=True).rename({"y": "lat", "x": "lon"})

    # convert -180--180 lons to 0--360
    #pop_da = population_tiff.assign_coords(
    #    lon=((population_tiff["lon"] + 360) % 360)
    #)

    # keep values strictly ascending
    #pop_da = pop_da.sortby("lon")
    #pop_da = pop_da.sortby("lat")

    # super important! gpwv4 nodatavals is -3.402823e+38 which will create weird values at grid interpolation
    #nodata = pop_da.attrs["_FillValue"]
    #pop_da = pop_da.where(pop_da != nodata)   # turn nodata into NaN

    # use xesmf for bilinear interpolation

    # define input (src) and output (dst) grids
    #src = xr.Dataset({"lat": pop_da.lat, "lon": pop_da.lon}) # input = pop_da
    #dst = xr.Dataset({"lat": ds.lat, "lon": ds.lon}) # output = ds that we want to match

    # define regridder object and apply to pop_da
    #regridder = xe.Regridder(src, dst, "bilinear")
    #pop_on_ds = regridder(pop_da)

    # adding population_density variable to ds
    #ds = ds.assign(population_density=pop_on_ds)

    # update ds.population_density attrs
    #ds["population_density"].attrs.update({
    #    "long_name": "Population density",
    #    "standard_name": "population_density",
    #    "units": "persons km-2",
    #})

    #return ds

def add_aux_data(ds):
    """Add grid weights and population density data to netcdf results 
    """
    ds = add_grid_weights(ds, grid="CESM2-WACCM6_0.9x1.25")
    #ds = add_population_density(ds)

    return ds

def get_mda8o3_climatology(
        config,
    ):
    
    path = (
        INTERIM_DATA_DIR 
        / config.institution_id
        / config.source_id
        / config.cmip_experiment_id
        / config.first_member
        / 'mda8o3'
    )
    file_name = (
            f"{config.institution_id}."
            f"{config.source_id}."
            f"{config.cmip_experiment_id}."
            f"{config.first_member}."
            "mda8o3."
            f"{config.year_range[0]}-{config.year_range[-1]}.nc"
        )
    ds = xr.open_dataset(path / file_name)
    ref_period = (ds['mda8o3']*1e9).sel(
            time=slice(f"{str(config.year_range[0])}-01-01", 
            f"{str(config.year_range[0]+9)}-12-31")).load()

    ref_monthly_clm = ref_period.groupby("time.month").mean("time")
    ref_monthly_clm.attrs["units"] = "ppbv"
    return ref_monthly_clm

def add_mda8o3_climatology(ds0, config):
    clm = get_mda8o3_climatology(config)

    assert clm.sizes["lat"] == ds0.sizes["lat"]
    assert clm.sizes["lon"] == ds0.sizes["lon"]

    clm2 = clm.assign_coords(lat=ds0.lat, lon=ds0.lon)
    ds0_clm = ds0.groupby("time.month") + clm2

    return ds0_clm