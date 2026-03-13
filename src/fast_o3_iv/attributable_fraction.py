import numpy as np
import xarray as xr

from fast_o3_iv.config import PROCESSED_DATA_DIR


def calc_af(osdma8, TMREL = 32.4, beta = np.log(1.063) / 10.0):
    """Calculate attributable fraction (%) for any relevant ozone fields present."""
    out = xr.Dataset()

    for var in ["y_true_bc", "y_true", "y_pred_bc", "y_pred"]:
        if var in osdma8:
            da = osdma8[var]
            excess = xr.where(da > TMREL, da - TMREL, 0.0)
            rr = np.exp(beta * excess)
            af = (rr - 1.0) / rr * 100
            out[f"{var}_af"] = af

    return out.sel(lat=slice(-60, 75))  # valid lats from RAMP obs

def create_af_save_path(config, model_key, mode):
    file_name = (
            f"{config.institution_id}."
            f"{config.source_id}."
            f"{config.cmip_experiment_id}."
            f"all_af."
            f"{config.variable_setname}."
            f"{mode}."
            f"{model_key}.nc"
        )

    file_path = (
            PROCESSED_DATA_DIR
            / "model_outputs"
            / "results"
            / mode
            / config.institution_id
            / config.source_id
            / config.cmip_experiment_id
            / "all_af"
            / config.variable_setname
            / model_key
            / file_name
        )

    file_path.parent.mkdir(parents=True, exist_ok=True)

    print(file_path, flush=True)

    return file_path