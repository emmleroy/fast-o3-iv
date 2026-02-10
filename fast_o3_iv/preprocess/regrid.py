import xarray as xr
import xesmf as xe

from fast_o3_iv.preprocess import calendar_tools


def regrid(ds, target_grid, convert_calendar=True):
    """Regrid an input (ds) to a target_grid using bilinear interpolation

    nice resource: https://climatedataguide.ucar.edu/climate-tools/regridding-overview
    """
    target_ds = target_grid.transpose("time", "lat", "lon")

    if "lon" not in ds.coords and "longitude" not in ds.coords:
        ds = ds.expand_dims(lon=target_ds["lon"]) 
        ds = ds.transpose("time", "lat", "lon")

    regridder = xe.Regridder(ds, target_ds, 'bilinear')
    ds_regridded = regridder(ds)

    if convert_calendar:
        original_units, original_calendar = calendar_tools.get_time_encoding(ds_regridded['time'])
        target_units, target_calendar = calendar_tools.get_time_encoding(target_ds['time'])
            
        if target_calendar != original_calendar:
            ds_regridded = ds_regridded.convert_calendar(target_calendar, dim='time')
            ds_regridded["time"].attrs = dict(target_ds["time"].attrs)
            ds_regridded["time"].encoding.update(target_ds["time"].encoding)

    new_attrs = dict(ds_regridded.attrs) # regrid_method = 'bilinear'
    old_attrs = dict(ds.attrs) # original attributes from input4MIPs for CESM
    ds_regridded.attrs = {**new_attrs, **old_attrs}

    return ds_regridded