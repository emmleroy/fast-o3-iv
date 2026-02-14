import os
from dataclasses import dataclass, field
from typing import Sequence

import xarray as xr

from fast_o3_iv.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from fast_o3_iv.preprocess.model_configs import MODEL_CONFIG
from fast_o3_iv.features.variable_setnames import VARIABLE_SET, pc_variable_sets
from fast_o3_iv.preprocess.process_loc_vars import ProcessLocVars

@dataclass
class CreateDataframe:
    """Create a pandas dataframe for testing/training from pre-processed xr.Datasets
    """
    input_source_id : str       # 'CESM2' or 'CESM2-WACCM6'
    experiment_id : str         # i.e. historical or ssp245
    training_member : str       # training ensemble member ID
    variant_label : str         # ensemble member ID
    variable_setname : str      # identifier for list of features to include
    year_range : Sequence[int]  # selected years (range(1975,2014)
    
    _training_set_created: bool = field(init=False, default=False, repr=False)

    def _get_input_filepath(self, variable):
        if (variable[-5:] != '_blur') and (variable[0:5] != 'mole_') :
            input_folder = f"{self.variant_label}"
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

    def _create_output_filepath(self):
        file_name = (
            f"{self.institution_id}."
            f"{self.input_source_id}."
            f"{self.experiment_id}."
            f"{self.variant_label}."
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
            / self.variant_label
            / self.variable_setname
            / file_name
        )
        return file_path
    
    def __post_init__(self): 
        """Compute derived attributes ("post initialization") """
        self._model_config = MODEL_CONFIG[self.input_source_id]
        self.institution_id = self._model_config["institution_id"]
        self.variable_set = VARIABLE_SET[self.variable_setname]
        self.has_chemistry = self._model_config["has_chemistry"]

    def create_dataset(self):
        """Take all variables in variable_set and create an xr.Dataset()

        not saved --> meant to be flexible so that each time we create a dataframe
        we can flexibly add/remove different features from the variable_set config
        """
        print("----------------")
        multivar_list = []
        template_lat = None

        for key, variables in self.variable_set.items():
            print(key + ": " + str(variables))
            if key == "target":
                for var in variables:
                    if self.has_chemistry:
                        print(f"Adding {var}...", flush=True)   
                        path = self._get_input_filepath(var)
                        singlevar_ds = xr.open_dataset(path)
                        multivar_list.append(singlevar_ds.transpose('time','lat','lon'))
                        #print(path)
                    else:
                        print(f"has_chemistry is False: not adding {var}")

            elif key in ("met", "chem"):
                for var in variables:
                    print(f"Adding {var}...", flush=True)   
                    path = self._get_input_filepath(var)
                    singlevar_ds = xr.open_dataset(path)

                    # to align pc vars later
                    if template_lat is None:
                        template_lat = singlevar_ds["lat"].values.copy()

                    multivar_list.append(singlevar_ds.transpose('time','lat','lon'))
                    #print(path)

            elif key == "pc_variables":
                for var in variables:
                    for subvars in pc_variable_sets[var]:
                        print(f"Adding {subvars}...", flush=True)
                    path = self._get_input_filepath(var)
                    singlevar_ds = xr.open_dataset(path)

                    # fix alignment issue
                    singlevar_ds = singlevar_ds.assign_coords(lat=template_lat)

                    multivar_list.append(singlevar_ds.transpose('time','lat','lon'))
                    #print(path)

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
        print("Adding month + year now...", flush=True)
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
                    topo_ds = xr.open_dataset(pipe.output_path)
                    
                    print(f"Adding {var}...")
                    multivar_ds[var] = topo_ds[var].broadcast_like(multivar_ds).transpose('time','lat','lon')
                
                elif var in ('lat', 'lon'):
                    print(f"{var} will be added later during conversion to dataframe")
        
        return multivar_ds
