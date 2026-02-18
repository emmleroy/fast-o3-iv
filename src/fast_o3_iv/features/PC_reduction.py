## take as input the ZG500 files in data/interim and return the hemispheric principal components ##

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import xarray as xr
from eofs.xarray import Eof

from fast_o3_iv.config import INTERIM_DATA_DIR
from fast_o3_iv.preprocess.model_configs import MODEL_CONFIG
from fast_o3_iv import utils

@dataclass 
class ApplyPCReduction:
    """Apply PCA for dimensionality reduction of a given field.

    note: this only takes pre-processed fields from data/interim
    """
    input_source_id : str       # 'CESM2' or 'CESM2-WACCM6'
    experiment_id : str         # i.e. historical or ssp245
    training_member : str       # training ensemble member ID
    var_input : str             # input variable name i.e. 'zg500'
    year_range : Sequence[int]  # selected years (range(1975,2015)
    k : int                     # number of principal components
    
    _is_fitted: bool = field(init=False, default=False, repr=False)

    def __post_init__(self): 
        """Compute derived attributes ("post initialization") """
        self.input_model_config = MODEL_CONFIG[self.input_source_id]
        self.institution_id = self.input_model_config["institution_id"]  
        if self.institution_id == "NCAR":
            self.training_source_id = "CESM2-WACCM6"
        else:
            raise ValueError(f"Only institution_id: {self.institution_id} support for now")
        
        self.var_output = f"hemis_{self.k}pcs_" + self.var_input
        self.training_path = self._get_filepath(self.training_source_id, self.var_input, self.training_member)        
        members_by_exp = self.input_model_config["members"]
        self.members = members_by_exp[self.experiment_id]

    def _get_filepath(self, source_id, variable, member):
        file_name = (
            f"{self.institution_id}."
            f"{source_id}."
            f"{self.experiment_id}."
            f"{member}."
            f"{variable}."
            f"{self.year_range[0]}-{self.year_range[-1]}.nc"
        )
        file_path = (
            INTERIM_DATA_DIR
            / self.institution_id
            / f"{source_id}"
            / self.experiment_id
            / f"{member}"
            / f"{variable}"
            / file_name
        )
        return file_path

    def _build_eof_solver(self, training_da):
        """Create an EOF solver (https://ajdawson.github.io/eofs/)
        using area-weighting --> sqrt(cos(lat))

        note: we take the *sqrt* of cos(lat) because the EOF solver must calculate the 
        covariance matrix (product XX^T)
        --> see more at: 
        https://www.ncl.ucar.edu/Document/Functions/Built-in/eofunc_n.shtml)

        note: center determines whether or not the time-mean is removed before
        analysis --> we set it to False because we first remove the climatological 
        monthly mean
        """
        # Define weights : sqrt(cos(lat))
        w_lat = np.sqrt(np.cos(np.deg2rad(training_da["lat"])))
        w2d, _ = xr.broadcast(w_lat, training_da["lon"])
        w2d = w2d.where(training_da.isel({"time": 0}).notnull())

        return Eof(training_da, weights=w2d, center=False)

    def fit(self):
        """Fit an EOF solver on the training array, separately for each hemisphere

        Doesn't return anything but it populates the 'ApplyPCReduction' object 
        with fitted parameters from the training array
        """
        # 0. get training da
        training_da = xr.open_dataarray(self.training_path)

        # 1. make sure training da has dims in order time/lat/lon
        da = training_da.transpose('time', 'lat', 'lon').load()

        # 2. create nh and sh mask
        self.lat = da['lat']
        self.lon = da['lon']

        # create nh/sh masks for use in _pcs_to_grid()
        self.nh_mask, _ = xr.broadcast(xr.where(self.lat >= 0, 1.0, 0.0), self.lon)
        self.sh_mask, _ = xr.broadcast(xr.where(self.lat < 0, 1.0, 0.0), self.lon)

        # 3. calculate anomalies relative to the training climatology
        reference_period = da.sel(
            time=slice(f"{str(self.year_range[0])}-01-01", 
            f"{str(self.year_range[0]+9)}-12-31")).load()
        
        self.training_clm_month = reference_period.groupby("time.month").mean("time")
        anomalies = da.groupby("time.month") - self.training_clm_month

        # 4. separate hemispheric anomalies & eof solvers
        self.nh_training_anm = anomalies.where(self.nh_mask, drop=True)
        self.sh_training_anm = anomalies.where(self.sh_mask, drop=True)

        self.eof_solver_nh = self._build_eof_solver(self.nh_training_anm)
        self.eof_solver_sh = self._build_eof_solver(self.sh_training_anm)

        self._is_fitted = True

    def _transform(self, variant_label):
        """project input_da fields onto the training EOF basis and return the 
        corresponding PC scores
        """
        # 0. make sure that training grid matches input grid and that fit() was called
        if not self._is_fitted:
            raise RuntimeError("Call `fit()` before `compute(input_da)`.")

        # filepaths paths for input member (to apply PC reduction to)
        # 1. get input_da and make sure input da has dims in order time/lat/lon
        input_path = self._get_filepath(self.input_source_id, self.var_input, variant_label)
        input_da = xr.open_dataarray(input_path)
        da = input_da.transpose('time', 'lat', 'lon').load()
        
        if da.sizes["lat"] != self.lat.size or da.sizes["lon"] != self.lon.size:
            raise ValueError("Input grid shape does not match training grid.")
        if not np.allclose(da["lat"].values, self.lat.values):
            raise ValueError("Input lat coords do not match training grid.")
        if not np.allclose(da["lon"].values, self.lon.values):
            raise ValueError("Input lon coords do not match training grid.")

        # 2. calculate anomalies relative to the training climatology
        # enforce identical coords otherwise anomalies calc will fail
        # note that this can cause the zg500 anomalies to have a nonzero mean
        da = da.assign_coords(lat=self.lat, lon=self.lon)
        anomalies = da.groupby("time.month") - self.training_clm_month

        # 3. separate hemispheric anomalies
        nh_input_anm = anomalies.where(anomalies["lat"] >= 0, drop=True)
        sh_input_anm = anomalies.where(anomalies["lat"] < 0, drop=True)

        # 4. separate hemispheric PCs
        pcs_nh = self.eof_solver_nh.projectField(
            nh_input_anm,
            neofs=self.k,
            eofscaling=1).assign_coords(
            mode=np.arange(1, self.k + 1)
        )
        pcs_sh = self.eof_solver_sh.projectField(
            sh_input_anm,
            neofs=self.k,
            eofscaling=1).assign_coords(
            mode=np.arange(1, self.k + 1)
        )

        # 5. use helper func to combine into one grid
        pcs_ds = self._pcs_to_grid(pcs_nh, pcs_sh)

        return pcs_ds

    def _pcs_to_grid(self, pcs_nh, pcs_sh):
        """helper function to convert hemispheric pcs onto a grid
        """
        pcs_grid_nh = (
            pcs_nh.expand_dims({'lat': self.lat, 'lon': self.lon})
            .transpose("time", "lat", "lon", "mode")
            * self.nh_mask
        )
        pcs_grid_sh = (
            pcs_sh.expand_dims({'lat': self.lat, 'lon': self.lon})
            .transpose("time", "lat", "lon", "mode")
            * self.sh_mask
        )
        pcs_grid = (pcs_grid_nh + pcs_grid_sh).transpose("time", "lat", "lon", "mode")

        labels = [f"pc{pc_num}_{self.var_input}" for pc_num in range(1, self.k + 1)]
        pcs_grid = pcs_grid.assign_coords(mode=("mode", labels))

        return pcs_grid.to_dataset(dim="mode")

    def output_path_for(self, variant_label):
        return self._get_filepath(self.input_source_id, self.var_output, variant_label)

    def _compute(self, variant_label):
        output_ds = self._transform(variant_label)
        output_path = self.output_path_for(variant_label)
        print(f"Ready to save PCs file --> {output_path}", flush=True) 
        return output_ds

    def compute_and_save(self, variant_label, save=True):
        """usage function
        """
        expected = 12 * len(self.year_range)
        output_path = self.output_path_for(variant_label)
        if utils.has_expected_data(output_path, expected):
            print(f"Skipping - {output_path} already exists and has {expected} timesteps.",
                  flush=True)
            return output_path

        ds = self._compute(variant_label)

        if save:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            ds.to_netcdf(output_path)
            print(f"Wrote final concatenated file --> {output_path}", flush=True)
            
        return output_path