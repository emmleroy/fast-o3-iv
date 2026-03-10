import os
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import xarray as xr
import pandas as pd

from fast_o3_iv.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from fast_o3_iv.preprocess.model_configs import MODEL_CONFIG
from fast_o3_iv.preprocess.process_loc_vars import ProcessLocVars
from fast_o3_iv.tabularize.variable_setnames import VARIABLE_SET, pc_variable_sets

@dataclass
class CreateDataframe:
    """Create a pandas dataframe for testing/training from pre-processed xr.Datasets
    """
    input_source_id : str       # 'CESM2' or 'CESM2-WACCM6'
    experiment_id : str         # i.e. historical or ssp245
    variable_setname : str      # identifier for list of features to include
    year_range : Sequence[int]  # selected years (range(1975,2014)
    
    _first_member_created: bool = field(init=False, default=False, repr=False)

    def __post_init__(self): 
        """Compute derived attributes ("post initialization") """
        self._model_config = MODEL_CONFIG[self.input_source_id]
        self.institution_id = self._model_config["institution_id"]
        self.variable_set = VARIABLE_SET[self.variable_setname]
        self.has_chemistry = self._model_config["has_chemistry"]
        members_by_exp = self._model_config["members"]
        self.members = members_by_exp[self.experiment_id]
        self.first_member = self.members[0]

    def _input_filepath(self, variable, variant_label):
        if (variable[-5:] != '_blur') and (variable[0:5] != 'mole_') :
            input_folder = f"{variant_label}"
        else:
            input_folder = "inputs"
        file_name = (
            f"{self.institution_id}."
            f"{self.input_source_id}."
            f"{self.experiment_id}."
            f"{input_folder}."
            f"{variable}."
            f"{self.year_range[0]}-{self.year_range[-1]}.nc"
        )
        file_path = (
            INTERIM_DATA_DIR
            / self.institution_id
            / self.input_source_id
            / self.experiment_id
            / input_folder
            / variable
            / file_name
        )
        return file_path

    def _get_output_filepath(self, variant_label):
        file_name = (
            f"{self.institution_id}."
            f"{self.input_source_id}."
            f"{self.experiment_id}."
            f"{variant_label}."
            f"{self.variable_setname}."
            "model_inputs."
            f"{self.year_range[0]}-{self.year_range[-1]}.parquet"
        )
        file_path = (
            PROCESSED_DATA_DIR
            / "model_inputs"
            / self.institution_id
            / self.input_source_id
            / self.experiment_id
            / variant_label
            / self.variable_setname
            / file_name
        )
        return file_path

    def _open_dataset(self, filepaths):
        """Open with xr.open_dataset with specific engine
        """ 
        engine="h5netcdf" # important!! to avoid binary incompatibility between numpy and netCDF

        ds_raw = xr.open_dataset(
            filepaths,
            engine=engine,
        )
        return ds_raw

    def _assemble_dataset(self, variant_label):
        """Take all variables in variable_set and create an xr.Dataset()

        not saved --> meant to be flexible so that each time we create a dataframe
        we can flexibly add/remove different features from the variable_set config
        """
        print("#######################################")
        print(f"Assembling dataset for {variant_label}")
        print("#######################################")
        print("----------------")
        multivar_list = []
        template_lat = None

        for key, variables in self.variable_set.items():
            print(key + ": " + str(variables))
            if key == "target":
                _, base_target, _ = self._target_spec()

                if self.has_chemistry:
                    print(f"Adding {base_target}...")
                    path = self._input_filepath(base_target, variant_label)
                    singlevar_ds = self._open_dataset(path)

                    if singlevar_ds[base_target].attrs.get("units") == "mol/mol":
                        singlevar_ds[base_target] = singlevar_ds[base_target] * 1e9
                        singlevar_ds[base_target].attrs["units"] = "ppbv"
                        print("converted mol/mol to ppbv")

                    multivar_list.append(singlevar_ds.transpose("time", "lat", "lon"))
                    print(path)
                else:
                    print(f"has_chemistry is False: not adding {base_target}")

            elif key in ("met", "chem"):
                for var in variables:
                    print(f"Adding {var}...")   
                    path = self._input_filepath(var, variant_label)
                    singlevar_ds = self._open_dataset(path)

                    # to align pc vars later
                    if template_lat is None:
                        template_lat = singlevar_ds["lat"].values.copy()

                    multivar_list.append(singlevar_ds.transpose('time','lat','lon'))
                    print(path)

            elif key == "pc_variables":
                for var in variables:
                    for subvars in pc_variable_sets[var]:
                        print(f"Adding {subvars}...")
                    path = self._input_filepath(var, variant_label)
                    singlevar_ds = self._open_dataset(path)

                    # fix alignment issue
                    singlevar_ds = singlevar_ds.assign_coords(lat=template_lat)

                    multivar_list.append(singlevar_ds.transpose('time','lat','lon'))
                    print(path)

            elif key == "time_variables":
                for var in variables:
                    print(f"Not adding {var} yet")
                    print(f"Month+Year will be added to merged dataset later")
            
            elif key == "location_variables":
                for var in variables:
                    print(f"Not adding {var} yet")
                    print(f"...will be added to merged dataset later")

            print("----------------")
        multivar_ds = xr.merge(multivar_list, join="exact", compat="override")
        print("Dataset created...")

        # Add month + year to all dataframes regardless of whether "month" is in the variable_set 
        # (the variable_set will be used by sklearn later)
        print("Adding month + year now...")
        multivar_ds["month"] = multivar_ds["time"].dt.month.broadcast_like(multivar_ds).transpose('time','lat','lon')
        multivar_ds["year"]  = multivar_ds["time"].dt.year.broadcast_like(multivar_ds).transpose('time','lat','lon')
        
        if 'month' in multivar_ds.coords: 
            multivar_ds = multivar_ds.reset_coords('month')
        
        if "lev" in multivar_ds.dims and multivar_ds.sizes["lev"] == 1:
            multivar_ds = multivar_ds.squeeze("lev", drop=True)
            multivar_ds = multivar_ds.drop_vars("lev", errors="ignore")

        if "location_variables" in self.variable_set.keys():
            for var in self.variable_set['location_variables']:
                if var in ('orog', 'sftlf'):
                    pipe = ProcessLocVars(
                        institution_id=self.institution_id
                    )
                    topo_ds = self._open_dataset(pipe.output_path)
                    
                    assert np.allclose(topo_ds.lat.values, multivar_ds.lat.values)
                    assert np.allclose(topo_ds.lon.values, multivar_ds.lon.values)

                    topo_fixed = topo_ds[var].assign_coords(
                        lat=multivar_ds.lat,
                        lon=multivar_ds.lon,
                    )

                    print(f"Adding {var}...")
                    multivar_ds[var] = topo_fixed.expand_dims(time=multivar_ds.time).transpose("time", "lat", "lon")
                
                elif var in ('lat', 'lon'):
                    print(f"{var} will be added later during conversion to dataframe")
        
        return multivar_ds

    def _target_spec(self):
        target_vars = self.variable_set.get("target", ())
        
        requested_target = target_vars[0]

        if requested_target.endswith("_anm"):
            base_target = requested_target[:-4]
            target_is_anomaly = True
        else:
            base_target = requested_target
            target_is_anomaly = False

        return requested_target, base_target, target_is_anomaly

    def anomaly_vars(self):
        """list of variables to convert into anomalies"""
        _, base_target, target_is_anomaly = self._target_spec()

        anomaly_vars_list = [
            *((base_target,) if target_is_anomaly else ()),
            *self.variable_set.get("met", ()),
            *self.variable_set.get("chem", ()),
        ]

        for pc_set in self.variable_set.get("pc_variables", ()):
            pc_entry = pc_variable_sets.get(pc_set, ())
            anomaly_vars_list.extend(pc_entry)

        return anomaly_vars_list

    def _rename_target_if_needed(self, ds):
        requested_target, base_target, target_is_anomaly = self._target_spec()

        if target_is_anomaly and base_target in ds.data_vars:
            ds = ds.rename({base_target: requested_target})

        return ds

    def create_first_member_ds(self):
        ds = self._assemble_dataset(self.first_member)

        requested_anomaly_vars = self.anomaly_vars()
        anomaly_vars = [v for v in requested_anomaly_vars if v in ds.data_vars]
        non_anomaly_vars = [v for v in ds.data_vars if v not in set(anomaly_vars)]

        # calculate and save anomalies relative to the first member climatology
        ref_period = ds[anomaly_vars].sel(
            time=slice(f"{str(self.year_range[0])}-01-01", 
            f"{str(self.year_range[0]+9)}-12-31")).load()
        self.first_member_clm_month = ref_period.groupby("time.month").mean("time")

        anomalies = ds[anomaly_vars].groupby("time.month") - self.first_member_clm_month
        if "month" in anomalies.coords:
            anomalies = anomalies.reset_coords("month", drop=True)
        non_anomalies = ds[non_anomaly_vars]
        
        ds_merged = xr.merge([anomalies, non_anomalies], join="exact", compat="override")
        ds_merged = self._rename_target_if_needed(ds_merged)

        self._first_member_created = True
        ds_merged.attrs["variant_label"] = self.first_member
        return ds_merged
        
    def create_ds(self, variant_label):
        ds = self._assemble_dataset(variant_label)

        requested_anomaly_vars = self.anomaly_vars()
        anomaly_vars = [v for v in requested_anomaly_vars if v in ds.data_vars]
        non_anomaly_vars = [v for v in ds.data_vars if v not in set(anomaly_vars)]

        # calculate and save anomalies relative to the first member climatology
        if self._first_member_created is False:
            self.create_first_member_ds()

        anomalies = ds[anomaly_vars].groupby("time.month") - self.first_member_clm_month
        if "month" in anomalies.coords:
            anomalies = anomalies.reset_coords("month", drop=True)
        non_anomalies = ds[non_anomaly_vars]
        
        ds_merged = xr.merge([anomalies, non_anomalies], join="exact", compat="override")
        ds_merged = self._rename_target_if_needed(ds_merged)
        ds_merged.attrs["variant_label"] = variant_label
        return ds_merged

    @staticmethod
    def convert_ds_to_df(ds):
        df = ds.to_dataframe().reset_index()

        # kill float wiggles with a given precision 
        df["lat"] = np.around(df["lat"].astype("float64"), 10)
        df["lon"] = np.around((df["lon"].astype("float64") % 360.0), 10)

        # remove time column
        df_out = df.drop(columns=['time'])

        # check nans
        for column in df_out.columns:   
            if df_out[column].isnull().values.any():
                raise ValueError("ERROR! nans in column: " + column)
        
        # check size
        expected_size = ds.sizes["time"] * ds.sizes["lat"] * ds.sizes["lon"]
        if len(df) != expected_size:
            raise ValueError("ERROR! ds and df are not the same size")

        return df_out

    def _compute(self, variant_label):
        self.output_path = self._get_output_filepath(variant_label)

        if variant_label == self.first_member:
            output_ds = self.create_first_member_ds()
        else:
            output_ds = self.create_ds(variant_label)

        output_df = self.convert_ds_to_df(output_ds)

        print(f"Ready to save df --> {self.output_path}")
        return output_df

    def _df_exists_and_has_expected_size(self):
        path = self.output_path
        expected_time = 12 * len(self.year_range)
        expected_lat = self._model_config["num_lat"]
        expected_lon = self._model_config["num_lon"]
        if not (os.path.exists(path) and os.path.getsize(path) > 0):
            return False
        try:
            df = pd.read_parquet(path)
            expected_rows = expected_time * expected_lat * expected_lon
            return len(df) == expected_rows
        except Exception:
            return False

    def compute_and_save(self, variant_label, save=True):
        self.output_path = self._get_output_filepath(variant_label)
        if self._df_exists_and_has_expected_size():
            print(f"Skipping - {self.output_path} already exists and has expected size.", 
                  flush=True)
            return self.output_path

        df = self._compute(variant_label)

        if save:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            df.to_parquet(self.output_path, compression="zstd", index=False)
            print(f"Wrote final file --> {self.output_path}", flush=True)
            
        return self.output_path
    