import glob
from dataclasses import dataclass
from typing import Sequence

import xarray as xr
from xarray.coders import CFDatetimeCoder

from fast_o3_iv.preprocess.calendar_tools import interp_each_month_linearly
from fast_o3_iv.preprocess import calendar_tools
from fast_o3_iv.config import EXTERNAL_DATA_DIR, INTERIM_DATA_DIR
from fast_o3_iv.preprocess.model_configs import MODEL_CONFIG 
from fast_o3_iv.preprocess import regrid

@dataclass
class ProcessChemData:
    """Pipeline to pre-process chem variables from input4CMIPs/input4CMIPs-cesm:
    
    1. Concatenate netCDF files
    2. Optionally collapse the sectoral dimension
    2. Interpolate linearly between anchor years for future scenario inputs
    3. Rename variables to CMIP6 convention (if using CESM)
    3. Move monthly timestamps to month-start
    4. Filter selected years
    5. Write a single netCDF file to data/interim
    """
    target_source_id : str      # 'CESM2' or 'CESM2-WACCM6'
    input_type : str            # either emissions or ghg_forcings
    experiment_id : str         # i.e. historical or ssp245
    var_original : str          # native variable name ('emiss_anthro')
    var_output : str            # output variable name i.e. 'CO_em_anthro'
    year_range : Sequence[int]  # selected years (range(1975,2014)

    def __post_init__(self): 
        """Compute derived attributes ("post initialization") """
        self._model_config = MODEL_CONFIG[self.target_source_id]
        self.input_id = self._model_config["input_id"]  
        self.institution_id = self._model_config["institution_id"]  
        members_by_exp = self._model_config["members"]
        self.members = members_by_exp[self.experiment_id]
        self._engine = "h5netcdf" if self.experiment_id == "historical" else "netcdf4"
        self.out_filename = (
            f"{self.institution_id}."
            f"{self.target_source_id}."
            f"{self.experiment_id}."
            "inputs."
            f"{self.var_output}."
            f"{self.year_range[0]}-{self.year_range[-1]}.nc"
        )
        self.out_path = (
            INTERIM_DATA_DIR
            / self.institution_id
            / self.target_source_id
            / self.experiment_id
            / "inputs"
            / self.var_output
            / self.out_filename
        )

    # Finding and opening files
    def _glob_all_filepaths(self):
        """Return a glob path for ALL files in data/external specified by the class
        (a glob path is a path with wildcards that can match multiple files/directories)
        """
        pattern = (
            EXTERNAL_DATA_DIR
            / self.input_id
            / self.input_type
            / self.experiment_id
            / self.var_output
            / "*.nc"
        )

        return sorted(glob.glob(str(pattern)))

    def _open_mfdataset(self, filepaths):
        """Open with xr.open_mfdataset, rename vars, then decode cftimes.
        """ 
        
        ds_raw = xr.open_mfdataset(filepaths,
                                    decode_times=False,
                                    engine=self._engine)

        # check for 3D vars
        if ('lev' in ds_raw.dims) or ('plev' in ds_raw.dims):
            raise NotImplementedError('only single-level variables support for now')

        # keep only the emissions we are interested in
        ds_keep = ds_raw[[self.var_original]]

        # inpsect metadata
        if ds_keep['time'].attrs['units'] == 'days since 0-1-1':
            # fix weird units (for GHGforcings)
            ds_keep['time'].attrs['units'] = 'days since 0001-01-01'

        # decode times
        time_coder = CFDatetimeCoder(use_cftime=True)
        
        return xr.decode_cf(ds_keep, decode_times=time_coder)

    def _open_file_and_rename_vars(self):
        filepaths = self._glob_all_filepaths()

        # 1. open file
        print(filepaths)
        ds0 = self._open_mfdataset(filepaths)

        # 2. fix month-start
        ds1 = calendar_tools.time_set_monthstart(ds0)
        ds1["time"].attrs = dict(ds0["time"].attrs)
        ds1["time"].encoding.update(ds0["time"].encoding)

        # 3. sum sectoral emissions
        ds2 = self._sum_sectoral_emissions(ds1)

        # 4. rename vars
        ds = ds2.rename_vars({self.var_original: self.var_output})
        ds[self.var_output].attrs.update({"original name": self.var_original})

        return ds

    def _sum_sectoral_emissions(self, ds):
        if "sector" in ds.dims:
            ds = ds.sum(dim="sector")
        return ds

    def _linearly_interpolate(self, ds):
        if self.experiment_id != 'historical':
            ds_interp = calendar_tools.interp_each_month_linearly(ds)
        else:
            ds_interp = ds
        return ds_interp

    def target_grid(self):
        grid_filename = (
            f"{self.institution_id}."
            f"{self.target_source_id}."
            f"{self.experiment_id}."
            f"{self.members[0]}."
            f"tas."
            f"{self.year_range[0]}-{self.year_range[-1]}.nc"
        )
        grid_path = (
            INTERIM_DATA_DIR
            / self.institution_id
            / self.target_source_id
            / self.experiment_id
            / self.members[0]
            / 'tas'
            / grid_filename
        )

        output_grid = xr.open_dataset(grid_path, engine= self._engine)

        return output_grid

    # Usage function
    def compute(self):

        # 0. open file
        ds0 = self._open_file_and_rename_vars()

        # 1. optionally linearly interpolate
        ds1 = self._linearly_interpolate(ds0)

        # 2. select years
        ds2 = ds1.sel(time=ds1.time.dt.year.isin(list(self.year_range)))

        # 3. regrid to target grid
        target_grid = self.target_grid()
        ds3 = regrid.regrid(ds2, target_grid, convert_calendar=True)


        print(f"Ready to save final concatenated file --> {self.out_path}", flush=True) 

        return ds3
        