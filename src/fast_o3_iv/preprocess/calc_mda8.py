from dataclasses import dataclass
import glob
import os
from typing import Sequence

import xarray as xr
from xarray.coders import CFDatetimeCoder

from fast_o3_iv.preprocess import calendar_tools
from fast_o3_iv.preprocess.model_configs import MODEL_CONFIG 
from fast_o3_iv.config import EXTERNAL_DATA_DIR, INTERIM_DATA_DIR
from fast_o3_iv import utils


@dataclass
class ProcessMDA8O3:
    """pipeline to pre-process hourly surface ozone files and produce monthly means of MDA8 O3
        var_name: 'O3_SRF' (cesm2) or 'sfo3' (cmip6)
        input_template: path to hourly ozone files and filename template
        output_template: where to store files
        """
    source_id : str             # name of climate model ('CESM2')
    experiment_id : str         # experiment_id ('historical')
    variant_label : str         # ensemble member ID
    var_original : str          # native variable name ('TS')
    year_range : Sequence[int]  # selected years (range(1975,2014)

    def __post_init__(self):
        self.var_output = "mda8o3"
        self._model_config = MODEL_CONFIG[self.source_id]
        self.institution_id = self._model_config["institution_id"]        
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
    def _glob_year_files(self, year):
        if self.source_id == "CESM2-WACCM6" and self.experiment_id == "historical":
            suffix = f"atm/hist/{self.variant_label}.cam.h4.{year}-*.nc"
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

    def _glob_all_filepaths(self):
        """Return a glob path for ALL files in data/external specified by the class
        (a glob path is a path with wildcards that can match multiple files/directories)
        """

        if (self.source_id == "CESM2-WACCM6") and (self.experiment_id == "historical"):
            suffix = f"atm/hist/{self.variant_label}.cam.h4.*.nc"
        else:
            suffix = f"{self.var_original}/*.nc"

        pattern = (
            EXTERNAL_DATA_DIR
            / self.institution_id
            / self.source_id
            / self.experiment_id
            / self.variant_label
            / suffix
        )
        
        return sorted(glob.glob(str(pattern)))

    def _open_subset(self, filepaths):
        """Open one (possibly multi-year) file at a time, decode CF times."""
        filepaths = [filepaths]

        ds0 = xr.open_mfdataset(
            filepaths,
            combine="by_coords",
            data_vars="minimal",
            coords="minimal",
            compat="override",
            decode_times=False,
            engine="netcdf4",
            parallel=True,
            chunks={"time": 240, "lat": -1, "lon": -1}
        )

        ds_keep = ds0[[self.var_original]]
        time_coder = CFDatetimeCoder(use_cftime=True)
        return xr.decode_cf(ds_keep, decode_times=time_coder)
    
    # Processing files
    def _process_subset(self, filepath_subset):

        print(f"working on {filepath_subset}, flush=true")
        print("opening file!", flush=True)
        ds0 = self._open_subset(filepath_subset)

        # 1. shift to local standard time
        print("shifting to LST!", flush=True)
        ds1 = calendar_tools.to_localtime(ds0)

        # 2. calculate mda8o3 on localtime data
        print("calculating mda8!", flush=True)
        mda8 = self._calc_mda8(ds1[self.var_original])

        # 3. calculate monthly mean of daily mda8o3
        print("resampling time!", flush=True) 
        monthly_mda8 = mda8.resample(time='MS').mean(keep_attrs=True)
        monthly_mda8.name = "mda8o3"

        # 4. update variable metadata 
        print("updating meta data!", flush=True) 
        var_units = ds0[self.var_original].attrs.get("units", "unknown")
        monthly_mda8.attrs.update({
            "long_name" : "Maximum daily 8-hour mean ozone (monthly mean of daily maxima)",
            "units" : var_units,
            "note" : "Calculated after shifting data to local standard time (no DST).",
        })

        # 5. update time metadata
        print("updating time encodig!", flush=True) 
        units, calendar = calendar_tools.get_time_encoding(ds0["time"])
        monthly_mda8["time"].encoding.update({
            "units": units,
            "calendar": calendar,
        })

        return monthly_mda8.to_dataset()

    @staticmethod
    def _calc_mda8(da):
        """calculate the maximum daily 8-h average of an xr.DataArray"""

        # 8-h running mean, valid if ≥6 of 8 hours present
        roll8 = da.rolling(time=8, min_periods=6).mean(keep_attrs=True)

        # label the window by its *start* hour
        roll8_start = roll8.shift(time=-7)

        # max of *all* 24 possible 8-h means in each calendar day
        mda8 = roll8_start.resample(time="1D").max(keep_attrs=True)

        return mda8

    # save files
    def _subset_outfile(self, year_or_year_group):
        tmp_out_filename = (
            f"{self.institution_id}."
            f"{self.source_id}."
            f"{self.experiment_id}."
            f"{self.variant_label}."
            f"{self.var_output}."
            "{year}.nc".format(year=year_or_year_group)
        )

        tmp_out_filepath = (
            INTERIM_DATA_DIR
            / self.institution_id
            / self.source_id
            / self.experiment_id
            / self.variant_label
            / self.var_output
            / tmp_out_filename
        )

        os.makedirs(os.path.dirname(tmp_out_filepath), exist_ok=True)
        return tmp_out_filepath

    def _save_file_subsets(self, ds, year_or_year_group):
        outfile = self._subset_outfile(year_or_year_group)
        print("saving to netcdf....", flush=True)
        ds.to_netcdf(outfile)
        print(f"Saved {year_or_year_group} -> {outfile}", flush=True)

    @staticmethod
    def _concat(yearly):
        return xr.concat(yearly, dim="time").sortby("time")

    def compute(self):
        """if CESM2-WACCM6 historical: process and save yearly files
        if CESM2-WACCM6 ssp scenario: process and save multi-year files

        note: need to eventually edit this for other CMIP6 models"""

        if self.source_id == "CESM2-WACCM6" and self.experiment_id == "historical":
            for year in self.year_range:
                outpath = self._subset_outfile(year)
                expected = 12
                if utils.has_expected_data(outpath, expected):
                    print(f"Skipping {year} --> file already exists and has {expected} timesteps.", flush=True)
                else:
                    print(f"Processing {year}", flush=True)
                    inpath = self._glob_year_files(year)
                    ds = self._process_subset(inpath)
                    self._save_file_subsets(ds, year)

        elif self.source_id == "CESM2-WACCM6" and self.experiment_id != "historical" and self.year_range == range(2015,2055):
            year_groups = [
                "2015-2024",
                "2025-2034",
                "2035-2044",
                "2045-2054"
            ]
            inpaths = self._glob_all_filepaths()
            for idx, year_group in enumerate(year_groups):
                outpath = self._subset_outfile(year_group)
                expected = 12 * 10
                if utils.has_expected_data(outpath, expected):
                    print(f"Skipping {year_group} --> file already exists and has {expected} timesteps.", flush=True)
                else:
                    print(f"Processing {year_group}", flush=True)
                    inpath = [inpaths[idx]]
                    ds = self._process_subset(inpath)
                    self._save_file_subsets(ds, year_group)
        else:
            raise NotImplementedError("only CESM2-WACCM6 historical and ssp245/ssp585 supported for now")

    def concat_saved(self):
        if self.source_id == "CESM2-WACCM6" and self.experiment_id == "historical":
            filepaths = [self._subset_outfile(year) for year in sorted(self.year_range)]
        elif self.source_id == "CESM2-WACCM6" and self.experiment_id != "historical" and self.year_range == range(2015,2055):
            year_groups = [
                "2015-2024",
                "2025-2034",
                "2035-2044",
                "2045-2054"
            ]
            filepaths = [self._subset_outfile(year_group) for year_group in year_groups]
        else:
            raise NotImplementedError("only CESM2-WACCM6 historical and ssp245/ssp585 supported for now")
        ds = xr.open_mfdataset(
            filepaths,
            combine="nested",
            concat_dim="time",
            data_vars="minimal",
            coords="minimal",
            compat="override",
            decode_times=False,
            engine="netcdf4",
            parallel=True,
            chunks={"time": 120},
        )
        
        expected = 12 * len(self.year_range)
        if utils.has_expected_data(self.out_path, expected):
            print(f"Skipping {self.out_path} --> file already exists and has {expected} timesteps.", flush=True)
        else:
            os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
            ds.to_netcdf(self.out_path)
            print(f"Wrote final concatenated file --> {self.out_path}", flush=True)