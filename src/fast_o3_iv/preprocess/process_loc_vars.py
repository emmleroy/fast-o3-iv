from dataclasses import dataclass
import os

import xarray as xr 

from fast_o3_iv import utils
from fast_o3_iv.config import EXTERNAL_DATA_DIR, INTERIM_DATA_DIR

@dataclass
class ProcessLocVars:
    institution_id : str          # i.e. NCAR
    topo_filename: str = (
        "fv_0.9x1.25_nc3000_Nsw042_Nrs008_Co060_Fi001_ZR_sgh30_24km_GRNL_c170103.nc"
    )

    def __post_init__(self):
        self.input_path = str(
            EXTERNAL_DATA_DIR
            / self.institution_id
            / "inputdata"
            / "topo"
            / self.topo_filename
        )
        
        self.output_path = str(
            INTERIM_DATA_DIR
            / self.institution_id
            / "inputdata"
            / "topo"
            / self.topo_filename
        )

    def _topo_original(self):
        return xr.open_dataset(self.input_path)

    def _topo_new(self):

        topo_orig = self._topo_original()

        g = 9.80665
        topo_orig['orog'] = topo_orig['PHIS'] / g
        topo_orig['orog'].attrs = {
            'long_name': 'surface altitude',
            'units': 'm',
            'original_var' : 'PHIS (surface geopotential) in m2/s2',
            'computed_as' : 'orog = PHIS/9.80665'
        }

        topo_orig['sftlf'] = topo_orig['LANDFRAC'] * 100
        topo_orig['sftlf'].attrs = {
            'long_name': 'percentage of the grid cell occupied by land (including lakes)',
            'units' : 'percentage',
            'original_var' : 'LANDFRAC (gridbox land fraction)',
            'computed_as' : 'sftlf = LANDFRAC * 100'
        }

        topo_new = topo_orig[['orog', 'sftlf']]

        print(f"Ready to save processed topo file --> {self.output_path}", flush=True) 

        return topo_new

    def compute_and_save(self, save=True):
        """usage function
        """
        if utils.exists(self.output_path):
            print(f"Skipping - file already exists.", 
                  flush=True)
            return self.output_path

        ds = self._topo_new()

        if save:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            ds.to_netcdf(self.output_path)
            print(f"Wrote processed topo file --> {self.output_path}", flush=True)

        return self.output_path