import numpy as np
import xarray as xr

from fast_o3_iv.config import PROCESSED_DATA_DIR


def calc_af(osdma8, TMREL = 32.4, beta = np.log(1.063) / 10.0):
    """Calculate attributable fraction (%)."""
    out = xr.Dataset()

    for var in ["y_true_bc", "y_true", "y_pred_bc", "y_pred"]:
        if var in osdma8:
            da = osdma8[var]
            excess = xr.where(da > TMREL, da - TMREL, 0.0)
            rr = np.exp(beta * excess)
            af = (rr - 1.0) / rr * 100
            out[f"{var}_af"] = af

    out = out.sel(lat=slice(-60, 75)) # valid lats from RAMP obs
    out.attrs.update({
            "af" : "attributable fraction, the percentage of deaths from chronic obstructive pulmonary disease (COPD) attributable to ozone pollution",
            "units" : "percent",
            "TMREL" : f"theoretical minimum risk exposure level, {TMREL} ppbv",
            "RR"  : "1.063 for each 10 ppbv increase in ozone above the TMREL",
            "beta": "log-linear RR slope = ln(1.063) / 10 ppbv",
            "health-relevant ozone metric" : "OSDMA8",
            "methodology" : "GBD 2019 methodology (Murray et al., 2020)"

        })
    return out

def return_af_path(mode, config, model_key):
    file_name = (
            f"{config.institution_id}."
            f"{config.source_id}."
            f"{config.cmip_experiment_id}."
            f"{mode}."
            f"{config.variable_setname}."
            f"{model_key}."
            f"attributable_fraction.bias_corrected.nc"
        )

    file_path = (
            PROCESSED_DATA_DIR
            / "model_outputs"
            / "derived_metrics"
            / config.institution_id
            / config.source_id
            / config.cmip_experiment_id
            / mode
            / config.variable_setname
            / model_key
            / file_name
        )

    file_path.parent.mkdir(parents=True, exist_ok=True)

    return file_path