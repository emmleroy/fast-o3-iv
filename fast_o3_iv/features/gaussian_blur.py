import os
from dataclasses import dataclass # convenient way to define a data class (https://www.datacamp.com/tutorial/python-data-classes)
from typing import Sequence # support for type hints

import scipy.ndimage as ndi
import xarray as xr

from fast_o3_iv.config import EXTERNAL_DATA_DIR, INTERIM_DATA_DIR
from fast_o3_iv.preprocess.model_configs import MODEL_CONFIG

@dataclass 
class ApplyGaussianBlur:
    """Apply anisotropic Gaussian blurring kernel to the emission fields (NOx or CO)

    note: this only takes model-specific inputs (emission files from input4MIPs that have 
    already been regridded and pre-processed to match a specific model)
    """
    source_id : str             # 'CESM2' or 'CESM2-WACCM6'
    experiment_id : str         # i.e. historical or ssp245
    var_input : str             # input variable name i.e. 'CO_em_anthro'
    year_range : Sequence[int]  # selected years (range(1975,2014)

    def __post_init__(self): 
            """Compute derived attributes ("post initialization") """
            self._model_config = MODEL_CONFIG[self.source_id]
            self.input_id = self._model_config["input_id"]  
            self.institution_id = self._model_config["institution_id"]  
            members_by_exp = self._model_config["members"]
            self.members = members_by_exp[self.experiment_id]
            self.input_filename = (
                f"{self.institution_id}."
                f"{self.source_id}."
                f"{self.experiment_id}."
                "inputs."
                f"{self.var_input}."
                f"{self.year_range[0]}-{self.year_range[-1]}.nc"
            )
            self.input_path = (
                INTERIM_DATA_DIR
                / self.institution_id
                / self.source_id
                / self.experiment_id
                / "inputs"
                / self.var_input
                / self.input_filename
            )
            self.var_output = self.var_input + '_blur'
            self.output_filename = (
                f"{self.institution_id}."
                f"{self.source_id}."
                f"{self.experiment_id}."
                "inputs."
                f"{self.var_output}."
                f"{self.year_range[0]}-{self.year_range[-1]}.nc"
            )
            self.output_path = (
                INTERIM_DATA_DIR
                / self.institution_id
                / self.source_id
                / self.experiment_id
                / "inputs"
                / self.var_output
                / self.output_filename
            )

    @staticmethod
    def apply_gaussian_blur(input_da, species):
        """Apply a Gaussian blurring kernel (filter) to an array with (time/lat/lon)
        for a specific emissions species (CO or NO)
        """

        if species not in {"NO", "CO"}:
            raise ValueError("species must be 'NO' or 'CO'")

        # 1x1.25 degree is ~100 km^2 at midlatitudes
        L = 1.0 if species == "NO" else 2.0 

        output_da = input_da.copy().load()
        output_da = output_da.transpose("time", "lat", "lon") # should already be in this order
        output_da.values = ndi.gaussian_filter(
            output_da.values,
            # sigma = standard deviation for Gaussian kernel
            # in array unit steps (degrees)
            sigma=(0.0, L, L),  # (time, lat, lon) 
            # mode = rule for what values exist outside the boundary
            # "nearest" means values outside are the nearest edge value
            # "wrap" means periodic boundary
            mode=("nearest", "nearest", "wrap"),
            # truncate = how far the Gaussian kernel extends before SciPy cuts it off
            # bigger truncate = closer to an infinite gaussian
            truncate=4.0, # = 4*L grid cells on each side
        )
        
        return output_da.to_dataset()

    def compute(self):
        input_ds = xr.open_dataset(self.input_path)
        output_ds = self.apply_gaussian_blur(input_ds[self.var_input],self.var_input[0:2])

        output_ds = output_ds.rename({
            self.var_input : self.var_output
        })

        print(f"Ready to save blurred emissions file --> {self.output_path}", flush=True) 

        return output_ds

    # Save / skip logic
    @staticmethod # static methods don't use self, acts like a utility under the class
    def _exists_and_has_n_timesteps(path, *, time_name="time", expected): # * so that following vars are key-word only
        if not (os.path.exists(path) and os.path.getsize(path) > 0):
            return False
        try:
            with xr.open_dataset(path, decode_cf=False, engine="netcdf4") as ds:
                return int(ds.sizes.get(time_name, 0)) == expected
        except Exception:
            return False

    def compute_and_save(self, save=True):
        expected = 12 * len(self.year_range)
        if self._exists_and_has_n_timesteps(self.output_path, expected=expected):
            print(f"Skipping - {self.output_path} already exists and has {expected} timesteps.", 
                  flush=True)
            return self.output_path

        ds = self.compute()

        if save:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            ds.to_netcdf(self.output_path)
            print(f"Wrote final concatenated file --> {self.output_path}", flush=True)
            
        return self.output_path


    
