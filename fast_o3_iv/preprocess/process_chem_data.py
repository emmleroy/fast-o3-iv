import glob, os
import xarray as xr
import numpy as np
import cftime
import datetime
import xesmf as xe
import dask

from src import tools, calendar_utils
from fast_o3_iv.preprocess.calendar_tools import interp_each_month_linearly

dask.config.set({"array.slicing.split_large_chunks": True})


class PreprocessChemData:

    """pipeline to pre-process MONTHLY files of input4MIPs (future scenario inputs must be interpolated)
        var_name: variable name
        input_template: path to raw monthly files and filename template
        output_template: where to store processed files
        """

    def __init__(self, var_name, years, input_template, output_template, grid_template):
        self.var = var_name
        self.years = years
        self.input_template = input_template
        self.output_template = output_template
        self.grid_template = grid_template

    def _open_selected_years(self):
        """open alls years of monthly files with raw time (no decoding), then decode to cftime"""
        
        # open files
        files = sorted(glob.glob(self.input_template))

        ds = xr.open_mfdataset(
                files,
                combine='nested', concat_dim='time',
                data_vars='minimal', coords='minimal', compat='override',
                decode_times=False,
                engine='netcdf4',                    parallel=True,
                chunks={"time": 240, "lat":-1, "lon":-1},
            )
    
        # inpsect metadata
        if ds['time'].attrs['units'] == 'days since 0-1-1':
            # fix weird units (for GHGforcings)
            ds['time'].attrs['units'] = 'days since 0001-01-01'

        # now decode with cftime (safe)
        ds = xr.decode_cf(ds, use_cftime=True)
        saved_attrs = ds.attrs
        # select variable of interest
        if (self.var or "").endswith("biomassburning"):
            var = self.var.removesuffix("_em_biomassburning")
            ds = ds[var].to_dataset()
        else:
            ds = ds[self.var].to_dataset()

        # linearly interpolate if emissions are decadal
        expected = np.array([2015, 2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100])
        decadal_years = np.unique(ds.time.dt.year.values)
        if np.array_equal(decadal_years, expected):
            # linearly interpolate between anchor years
            ds_interpolated = interp_each_month_linearly(ds)
            ds_interpolated["time"].attrs = dict(ds["time"].attrs)
            ds_interpolated["time"].encoding.update(ds["time"].encoding)

        else:
            ds_interpolated = ds


        # select time period of interest
        ds_interpolated = ds_interpolated.sel(time=ds_interpolated.time.dt.year.isin(self.years))
        ds_interpolated.attrs = dict(saved_attrs)
        return ds_interpolated


    def _regrid(self, ds):
        """regrid input4MIPs to match target model"""
        grid_file = self.grid_template
        target_ds = xr.open_dataset(grid_file, decode_times=True)
        target_ds = target_ds.transpose("time", "lat", "lon")
        if "lon" not in ds.coords and "longitude" not in ds.coords:
            ds = ds.expand_dims(lon=target_ds["lon"]) 
            ds = ds.transpose("time", "lat", "lon")

        regridder = xe.Regridder(ds, target_ds, 'bilinear')
        ds_regridded = regridder(ds)

        
        units_original, calendar_original = self.get_time_encoding(ds_regridded['time'])
        units, calendar = self.get_time_encoding(target_ds['time'])

        if calendar != calendar_original:
            ds_regridded['time'] = target_ds['time']
            ds_regridded["time"].attrs = dict(target_ds["time"].attrs)
            ds_regridded["time"].encoding.update(target_ds["time"].encoding)

        return ds_regridded

    def compute(self):

        # open selected data
        ds00 = self._open_selected_years()

        # check for 3D vars
        if ('lev' in ds00.dims) or ('plev' in ds00.dims):
            raise NotImplementedError('only single-level variables support for now')
        
        # adjust calendar
        ds0 = calendar_utils.snap_monthly_to_month_start(ds00)

        # sum sectoral emissions
        if "sector" in ds0.dims: 
            ds = ds0.sum(dim='sector', keep_attrs=True)
        else:
            ds = ds0

        # regrid
        ds_regridded = self._regrid(ds)
        ds_regridded.attrs.update(dict(ds.attrs))
        return ds_regridded
    
    @staticmethod
    def _exists_and_has_480_timesteps(path: str, *, time_name="time", expected=480):
        if not (os.path.exists(path) and os.path.getsize(path) > 0):
            return False
        try:
            with xr.open_dataset(path, decode_cf=False, engine="netcdf4") as ds:
                return int(ds.sizes.get(time_name, 0)) == expected
        except Exception:
            return False

    def compute_and_save(self): 
        outdir = self.output_template

        if self._exists_and_has_480_timesteps(outdir):
            print(f"Skipping - file already exists and has 480 timesteps.", flush=True)
        else:
            ds = self.compute()
            os.makedirs(os.path.dirname(outdir), exist_ok=True)
            ds.to_netcdf(outdir) 
            print(f"Wrote final concatenated file --> {outdir}", flush=True)
