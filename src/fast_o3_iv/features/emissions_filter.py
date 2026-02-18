import os
from dataclasses import dataclass 
from typing import Sequence, Literal

import numpy as np
import xarray as xr
from scipy import ndimage as ndi

from fast_o3_iv.config import EXTERNAL_DATA_DIR, INTERIM_DATA_DIR
from fast_o3_iv.preprocess.model_configs import MODEL_CONFIG
from fast_o3_iv import utils

@dataclass 
class ApplyEmissionsFilter:
    """Apply blurring kernel to the emission fields (NOx or CO)

    note: this only takes model-specific inputs (emission files from input4MIPs that have 
    already been regridded and pre-processed to match a specific model)
    """
    source_id : str             # 'CESM2' or 'CESM2-WACCM6'
    experiment_id : str         # i.e. historical or ssp245
    var_input : str             # input variable name i.e. 'CO_em_anthro'
    year_range : Sequence[int]  # selected years (range(1975,2014)
    method: Literal["isotropic", "downwind"] # literals restrict type hints to specific constants
    variant_label: str | None = None

    def __post_init__(self):
        """Compute derived attributes ("post initialization") """
        if self.method != "isotropic" and not self.variant_label:
            raise ValueError("variant_label is required unless method == 'isotropic'.")
        self._model_config = MODEL_CONFIG[self.source_id]
        self.input_id = self._model_config["input_id"]  
        self.institution_id = self._model_config["institution_id"] 

        if self.method == "isotropic":
            self.var_output = self.var_input + '_blur'
        else:
            self.var_output = self.var_input + '_dwblur' # downwind blur
            members_by_exp = self._model_config["members"]
            self.members = members_by_exp[self.experiment_id]

        self.input_path = self._get_filepath(self.var_input)
        self.output_path = self._get_filepath(self.var_output)

    def _get_filepath(self, variable):
        if variable in ["ua", "va"]:
            input_folder = f"{self.variant_label}"
        elif (self.method=='downwind') and (variable==self.var_output):
            input_folder = f"{self.variant_label}"
        else:
            input_folder = "inputs"
        file_name = (
            f"{self.institution_id}."
            f"{self.source_id}."
            f"{self.experiment_id}."
            f"{input_folder}."
            f"{variable}."
            f"{self.year_range[0]}-{self.year_range[-1]}.nc"
        )
        file_path = (
            INTERIM_DATA_DIR
            / self.institution_id
            / self.source_id
            / self.experiment_id
            / input_folder
            / variable
            / file_name
        )
        return file_path

    @staticmethod
    def apply_isotropic_blur(input_da, species):
        """Apply a Gaussian blurring kernel (filter) to an array with (time/lat/lon)
        for a specific emissions species (CO or NO)
        """

        if species not in {"NO", "CO"}:
            raise ValueError("species must be 'NO' or 'CO'")

        # 1x1.25 degree is ~100 km^2 at midlatitudes
        L = 1.0 if species == "NO" else 2.0 # (note: 3.0 in downwind blur)

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

    @staticmethod
    def _shift_map(field, dy, dx):
        """Helper function to shift a 2D field by (dy, dx):
        - lon --> wraps (periodic)
        - lat --> uses nearest edge value

        note: each kernel offset (dy, dx) needs the neighbor value at that offset for every cell.
        shifting the whole map is a vectorized way to get those neighbor values.
        """
        # 1. longitude shift
        lon_shifted_array = np.roll(field, shift=dx, axis=1)  # shift longitude (axis=1 for 2D field) by dx using wrapping
        
        # 2. latitude shift
        n_lat = field.shape[0] # number of latitudes
        target_lat_idx = np.arange(n_lat)# indices of the output lats
        source_lat_idx = target_lat_idx - dy # apply requested vertical shift by subtracting dy
        # force every latitude index to stay within valid bounds [0, n_lat-1]
        # so if an index is <0 after shifting, it becomes 0
        # if an index is > n_lat-1 after shifting, it becomes n_lat-1 
        source_lat_idx = np.clip(source_lat_idx, 0, n_lat - 1)  # "clamp to edges"

        # 3. build lat-shifted array after lon-shifted array
        latlon_shifted_array = lon_shifted_array[source_lat_idx, :]

        return latlon_shifted_array

    def _apply_downwind_blur_one_month(self, input_da0, ua_da0, va_da0, species):

        # 0. convert emissions inputs and wind inputs to np.arrays
        field = np.asarray(input_da0, dtype=np.float64)
        u = np.asarray(ua_da0, dtype=np.float64)
        v = np.asarray(va_da0, dtype=np.float64)

        # 1. define parameters for the downwind kernel
        L = 1.0 if species == "NO" else 3.0  # scaling for different species
        sigma_cross = 0.8 * L                # gaussian width in cross-wind direction (standard deviation)
        sigma_upwind = 0.6 * L               # gaussian width in up-wind direction (standard deviation)
        L_downwind = 2.5 * L                 # e-folding length in down-wind direction (distance where the value becomes 1/e ≈ 37% of the peak)
        original_frac = 0.5                  # fraction of the original emission field retained after blur (new_field = original_frac * original + (1-original_frac) * blurred)
        calm_thresh = 0.5                    # threshold for calm winds underwhich we just use an isotropic kernel
        r = int(np.ceil(4 * L_downwind))     # kernel radius = truncated to next whole integer that is at most (4 * L_down)

        speed = np.hypot(u, v)
        calm = (speed < calm_thresh)     

        # 2. calculate the wind direction normalized to unit length
        # ux = u / speed, uy = v / speed
        ux = np.divide(u, speed, out=np.zeros_like(u), where=~calm)
        uy = np.divide(v, speed, out=np.zeros_like(v), where=~calm)

        # 3. initialize the weighted average numerator and denominator
        # so that we can normalize by total kernel weight later

        # define emissions NaN mask
        field_valid = np.isfinite(field)
        field_filled = np.where(field_valid, field, 0.0)

        weighted_value_sum = np.zeros_like(field_filled, dtype=np.float64)
        weight_sum = np.zeros_like(field_filled, dtype=np.float64)

        # 4. loop through each kernel index
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                # retrieve the dy,dx neighbor by shifting the field
                shifted_field = self._shift_map(field_filled, dy, dx)

                # coordinate rotation: convert dx,dy to wind-aligned coordinates
                along_wind = dx * ux + dy * uy # distance along the wind
                cross_wind = -dx * uy + dy * ux # distance across the wind
                # note: if wind was pointing east, (ux,uy) = (1,0)
                # so then along_wind = dx, cross_wind = dy

                # Directional (wind-aligned) weight components
                w_cross = np.exp(-0.5 * (cross_wind / sigma_cross) ** 2) # gaussian with max at cross_wind = 0
                w_upwind = np.exp(-0.5 * (np.minimum(along_wind, 0.0) / sigma_upwind) ** 2) # gaussian upwind where along_wind<0
                w_downwind = np.exp(-np.maximum(along_wind, 0.0) / L_downwind) # exponential decay downwind where along_wind>0
                w_dir = w_cross * w_upwind * w_downwind

                # For calm winds, fall back to a simpler isotropic Gaussian
                w_iso = np.exp(-0.5 * ((dx / L) ** 2 + (dy / L) ** 2))

                # Choose directional vs isotropic per grid cell
                w = np.where(calm, w_iso, w_dir)

                weighted_value_sum += w * shifted_field
                weight_sum += w

        blurred = np.divide(
            weighted_value_sum,
            weight_sum,
            out=np.full_like(weighted_value_sum, np.nan),
            where=weight_sum > 0,
        )

        total = original_frac * field_filled + (1.0 - original_frac) * blurred
        return total.astype(input_da0.dtype, copy=False)

    def apply_downwind_blur(self, input_da, species):
        """Apply a non-Gaussian blurring kernel (filter) to an array with (time/lat/lon)
        for a specific emissions species (CO or NO)

        Kernel: exponential in the downwind direction, gaussian across wind and upwind

        a bit like the pyELQ gaussian plume code: 
        https://github.com/sede-open/pyELQ/blob/main/src/pyelq/dispersion_model/gaussian_plume.py
        """

        if species not in {"NO", "CO"}:
            raise ValueError("species must be 'NO' or 'CO'")
        
        # 1. open the emissions input and wind fields
        output_da0 = input_da.copy().load()
        output_da0 = output_da0.transpose("time", "lat", "lon")
        output_da = output_da0.fillna(0) # needed for openburning emissions

        ua_ds = xr.open_dataset(self._get_filepath('ua'))
        ua_da = ua_ds['ua'].transpose("time", "lat", "lon")

        va_ds = xr.open_dataset(self._get_filepath('va'))
        va_da = va_ds['va'].transpose("time", "lat", "lon")

        # 2. apply blurring month by month
        blurred = xr.apply_ufunc(
            self._apply_downwind_blur_one_month,
            output_da,
            ua_da,
            va_da,
            input_core_dims=[["lat", "lon"], ["lat", "lon"], ["lat", "lon"]],
            output_core_dims=[["lat", "lon"]],
            kwargs={"species": species},
            vectorize=True,
            dask="parallelized",
            output_dtypes=[output_da.dtype],
        )

        return blurred.to_dataset(name=self.var_input)

    def _compute(self):
        input_ds = xr.open_dataset(self.input_path)

        if self.method == 'isotropic':
            output_ds = self.apply_isotropic_blur(input_ds[self.var_input], self.var_input[0:2])
        elif self.method == 'downwind':
            output_ds = self.apply_downwind_blur(input_ds[self.var_input], self.var_output[0:2])
        else:
            raise NotImplementedError("only 'isotropic' or 'downwind' methods supported")

        output_ds = output_ds.rename({
            self.var_input : self.var_output
        })

        print(f"Ready to save blurred emissions file --> {self.output_path}", flush=True) 

        return output_ds

    def compute_and_save(self, save=True):
        """usage function
        """
        expected = 12 * len(self.year_range)
        if utils.has_expected_data(self.output_path, expected):
            print(f"Skipping - {self.output_path} already exists and has {expected} timesteps.", 
                  flush=True)
            return self.output_path

        ds = self._compute()

        if save:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            ds.to_netcdf(self.output_path)
            print(f"Wrote final concatenated file --> {self.output_path}", flush=True)

        return self.output_path