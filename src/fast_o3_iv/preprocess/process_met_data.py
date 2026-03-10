import glob
import os
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np
import xarray as xr
from xarray.coders import CFDatetimeCoder
from geocat.comp.interpolation import interp_hybrid_to_pressure

from fast_o3_iv.preprocess import calendar_tools
from fast_o3_iv.preprocess.model_configs import MODEL_CONFIG 
from fast_o3_iv.config import EXTERNAL_DATA_DIR, INTERIM_DATA_DIR
from fast_o3_iv import utils


@dataclass
class ProcessMetData:
    """Pipeline to pre-process met variables from CESM/CMIP:
    
    1. Concatenate netCDF files
    2. Rename variables to CMIP6 convention (if using CESM)
    3. Optionally select a vertical level (surface or interpolate to a pressure level)
    4. Move monthly timestamps to month-start
    5. Filter selected years
    6. Write a single netCDF file to data/interim
    """
    source_id : str             # name of climate model ('CESM2')
    experiment_id : str         # experiment_id ('historical')
    variant_label : str         # ensemble member ID
    var_original : str          # native variable name ('TS')
    year_range : Sequence[int]  # selected years (range(1975,2014)
    level: Optional[Union[str, float]] = None # can be : None, "surface" or numeric hPa

    def __post_init__(self): 
        """Compute derived attributes ("post initialization") """

        # allow numeric pressure interpolation only for CESM2-WACCM6 historical
        #if self.level not in (None, "surface") and not (
        #    self.source_id == "CESM2-WACCM6" and self.experiment_id == "historical"
        #):
        #    raise ValueError(
        #        "Pressure-level interpolation is only supported for CESM2-WACCM6 historical for now."
        #    ) # raise error for now --> need to fix this

        self.model_config = MODEL_CONFIG[self.source_id]
        self.variable_map = self.model_config.get("variable_map")
        self.institution_id = self.model_config["institution_id"]  
        self.var_output = self.variable_map.get(self.var_original, self.var_original)
        if self.level not in (None, "surface"):
            self.level = float(self.level)
        self.out_filename = (
            f"{self.institution_id}."
            f"{self.source_id}."
            f"{self.experiment_id}."
            f"{self.variant_label}."
            f"{self.var_output}."
            f"{self.year_range[0]}-{self.year_range[-1]}.nc"
        )
        self.out_path = (
            INTERIM_DATA_DIR
            / self.institution_id
            / self.source_id
            / self.experiment_id
            / self.variant_label
            / self.var_output
            / self.out_filename
        )

    # Finding and opening files
    def _glob_all_filepaths(self, variable):
        """Return a glob path for ALL files in data/external specified by the class
        (a glob path is a path with wildcards that can match multiple files/directories)
        """

        if (self.source_id == "CESM2-WACCM6") and (self.experiment_id == "historical"):
            suffix = f"atm/hist/{self.variant_label}.cam.h0.*.nc"
        else:
            suffix = f"{variable}/*.nc"

        pattern = (
            EXTERNAL_DATA_DIR
            / self.institution_id
            / self.source_id
            / self.experiment_id
            / self.variant_label
            / suffix
        )
        
        return sorted(glob.glob(str(pattern)))

    def _glob_year_files(self, year):
        """Return a glob path for yearly files in data/external specified by the class
        that match the selected year.

        note! CESM2_WACCM6 met_data has a specific format in data/external on Svante
        --> can be updated when/if that output is standardized and published
        """
        if self.source_id == "CESM2-WACCM6" and self.experiment_id == "historical":
            suffix = f"atm/hist/{self.variant_label}.cam.h0.{year}*.nc"
        else:
            suffix = f"{self.var_original}/*{year}*.nc"

        pattern = (
            EXTERNAL_DATA_DIR
            / self.institution_id
            / self.source_id
            / self.experiment_id
            / self.variant_label
            / suffix
        )

        return sorted(glob.glob(str(pattern)))

    def _required_vars(self):
        """If level is None or 'surface', only return var_original
        otherwise, include extra vars needed for pressure interpolation: 
        
        for CESM2-WACCM6 historical: ("PS", "hyam", "hybm", "P0")
        for others: return all vars (for now)

        """
        if self.level in (None, "surface"):
            return [self.var_original]

        return [self.var_original, "PS", "hyam", "hybm", "P0"]


    def _open_mfdataset(self, filepaths):
        """Open with xr.open_mfdataset, keep only required vars, then decode cftimes.
        """ 
        if self.source_id == 'CESM2-WACCM6':
            engine="netcdf4"
        elif (self.source_id == 'CESM2') and (self.experiment_id!='historical'):
            engine="netcdf4"
            print("using netcdf", flush=True)
        else:
            engine="h5netcdf" # important!! to avoid binary incompatibility between numpy and netCDF

        ds_raw = xr.open_mfdataset(
            filepaths,
            combine="nested",
            concat_dim="time",
            compat="no_conflicts",
            data_vars="minimal",
            decode_times=False,
            engine=engine,
        )
        
        required_vars = self._required_vars()

        # some formats have PS saved in a different file
        need_ps = (
            required_vars is not None
            and "PS" in required_vars
            and "PS" not in ds_raw.data_vars
        )

        if need_ps:
            filepaths_ps = self._glob_all_filepaths("PS")  # make sure this signature exists
            if not filepaths_ps:
                raise FileNotFoundError("PS files not found")

            ds_ps = xr.open_mfdataset(
                filepaths_ps,
                combine="nested",
                concat_dim="time",
                compat="no_conflicts",
                data_vars="minimal",
                decode_times=False,
                engine=engine,
            )[["PS"]]

            ds_raw = xr.merge([ds_raw, ds_ps], join="exact", compat="override")

        ds_keep = ds_raw if required_vars is None else ds_raw[required_vars]

        time_coder = CFDatetimeCoder(use_cftime=True)
        return xr.decode_cf(ds_keep, decode_times=time_coder)

    def _open_selected_years_and_levels(self):
        """If num files >= 40, assume we have annual files and open one by one
        """
        filepaths = self._glob_all_filepaths(self.var_original)
        if not filepaths:
            raise FileNotFoundError(f"No files matched: {filepaths}")

        if len(filepaths) >= 40:
            parts = []
            for y in self.year_range: # only select annual files in year_range 
                year_file = self._glob_year_files(y)
                print(f"opening {y}", flush=True)
                # 1. open file
                ds_y = self._open_mfdataset(year_file)
                # 2. fix month-start
                ds_y2 = calendar_tools.time_set_monthstart(ds_y)
                ds_y2["time"].attrs = dict(ds_y["time"].attrs)
                ds_y2["time"].encoding.update(ds_y["time"].encoding)
                # 3. select vertical level
                ds_y2 = self._select_vertical_lev(ds_y2)
                parts.append(ds_y2[self.var_original])
            return xr.concat(parts, dim="time").sortby("time").to_dataset()
        else:
            # 1. open file
            ds = self._open_mfdataset(filepaths)
            # 2. fix month-start
            ds2 = calendar_tools.time_set_monthstart(ds)
            ds2["time"].attrs = dict(ds["time"].attrs)
            ds2["time"].encoding.update(ds["time"].encoding)
            # 3. select years
            ds2 = ds2.sel(time=ds2.time.dt.year.isin(list(self.year_range)))
            # 4. select vertical level
            ds2 = self._select_vertical_lev(ds2)
            return ds2[self.var_original].to_dataset()

    # Reduce the vertical dimension
    def _select_vertical_lev(self, ds):            
        """Ensures ds contains var_input as either:
          - unchanged (level=None)
          - surface-selected (level="surface")
          - interpolated to one pressure level (level=numeric hPa)
        """
        if self.level is None:
            return ds

        if self.level == "surface":
            if "lev" not in ds.dims:
                # treat as already single-level
                return ds

            positive = ds["lev"].attrs.get("positive", "").lower()
            if positive == "down":
                ds2 = ds.isel(lev=-1, drop=True)
            elif positive == "up":
                ds2 = ds.isel(lev=0, drop=True)
            else:
                raise NotImplementedError(
                    'need ds.lev.attrs["positive"] to be "up" or "down"')

            # Drop optional CESM-specific extras if they exist
            drop_vars = [v for v in ["ilev", "zlon"] if v in ds2.variables]
            ds2 = ds2.drop_vars(drop_vars, errors="ignore")
            for d in ["ilev", "zlon"]:
                if d in ds2.dims:
                    ds2 = ds2.drop_dims(d)
            return ds2

        # Convert numeric pressure level in hPa -> Pa
        target_pa = np.array([float(self.level) * 1e2])

        var = ds[self.var_original]
        var_level = interp_hybrid_to_pressure(
            data=var,
            ps=ds["PS"],
            hyam=ds["hyam"],
            hybm=ds["hybm"],
            p0=ds["P0"],
            new_levels=target_pa,
            method="linear",
        )
        # Make sure the result keeps the expected variable name
        var_level = var_level.rename(self.var_original)
        return var_level.to_dataset()

    def _drop_single_vertical_dim(self, ds):
        # If we ended up with a single lev/plev, drop it safely
        for dim in ("plev", "lev"):
            if dim in ds.dims:
                if ds.sizes.get(dim, 0) == 1:
                    ds = ds.isel({dim: 0}, drop=True)
                else:
                    raise NotImplementedError(
                        f"{dim} has size {ds.sizes[dim]} (expected 1).")
        return ds
  
    def _compute(self):
        ds = self._open_selected_years_and_levels()
        ds = self._drop_single_vertical_dim(ds)

        # Rename to desired output name
        if self.var_original != self.var_output:
            ds = ds.rename_vars({self.var_original: self.var_output})
        ds[self.var_output].attrs.update({"original name": self.var_original})

        # Filter requested years (noteL isn't this repeated?)
        ds2 = ds.sel(time=ds.time.dt.year.isin(self.year_range))
        ds2.attrs.update(dict(ds.attrs))

        ds2 = ds2.load()  # faster final write for 2D output

        print(f"Ready to save final concatenated file --> {self.out_path}", flush=True) 

        return ds2

    def compute_and_save(self, save=True):
        """Usage function
        """
        expected = 12 * len(self.year_range)
        if utils.has_expected_data(self.out_path, expected):
            print(f"Skipping {self.out_path} already exists and has {expected} timesteps.",
                  flush=True)
            return self.out_path

        ds = self._compute()

        if save:
            os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
            ds.to_netcdf(self.out_path)
            print(f"Wrote final concatenated file --> {self.out_path}", flush=True)
        return self.out_path
