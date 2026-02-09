import xarray as xr
import xesmf as xe

from fast_o3_iv.preprocess import calendar_tools


def regrid(ds, target_grid):
    """Regrid an input (ds) to a target_grid using bilinear interpolation

    nice resource: https://climatedataguide.ucar.edu/climate-tools/regridding-overview
    """
    target_ds = target_grid.transpose("time", "lat", "lon")

    if "lon" not in ds.coords and "longitude" not in ds.coords:
        ds = ds.expand_dims(lon=target_ds["lon"]) 
        ds = ds.transpose("time", "lat", "lon")

    regridder = xe.Regridder(ds, target_ds, 'bilinear')
    ds_regridded = regridder(ds)

    units_original, calendar_original = calendar_tools.get_time_encoding(ds_regridded['time'])
    units, calendar = calendar_tools.get_time_encoding(target_ds['time'])
        
    if calendar != calendar_original:
        ds_regridded['time'] = target_ds['time']
        ds_regridded["time"].attrs = dict(target_ds["time"].attrs)
        ds_regridded["time"].encoding.update(target_ds["time"].encoding)

    return ds_regridded