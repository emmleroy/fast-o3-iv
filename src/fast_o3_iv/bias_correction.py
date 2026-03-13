from fast_o3_iv.config import PROCESSED_DATA_DIR

def get_bc_scaling_factor(model_osmda8_ds, obs_osdma8_ds):
    """get bias correction scaling factor"""
    
    baseline_years = slice("1990", "2014")

    obs = obs_osdma8_ds['ozone'] # osdma8

    ytrue = model_osmda8_ds["y_true"]
    ytrue_forced = ytrue.mean("ensemble_member") # ensemble means

    # baseline climatologies
    obs_base = obs.sel(time=baseline_years).mean("time")
    mod_base = ytrue_forced.sel(time=baseline_years).mean("time")

    # ratio-based bias-correction factor
    scaling_factor = obs_base / mod_base

    return scaling_factor.transpose("lat", "lon"), obs_base, mod_base

def bias_correct(model_osmda8_ds, scaling_factor):
    model_osmda8_bc_ds = model_osmda8_ds.copy()

    for var in ["y_true", "y_pred"]:
        if var in model_osmda8_ds:
            da = model_osmda8_ds[var]
            forced = da.mean(dim="ensemble_member")
            forced_bc = scaling_factor * forced
            da_bc = forced_bc + (da - forced)
            model_osmda8_bc_ds[f"{var}_bc"] = da_bc

    if "y_pred" not in model_osmda8_ds:
        raise KeyError("model_osmda8_ds must contain 'y_pred'")

    model_osmda8_bc_ds["scaling_factor"] = scaling_factor
    return model_osmda8_bc_ds.transpose("time", "lat", "lon", "ensemble_member")

def create_osdma8_bc_save_path(config, model_key, mode):
    file_name = (
            f"{config.institution_id}."
            f"{config.source_id}."
            f"{config.cmip_experiment_id}."
            f"all_osdma8_biascorrected."
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
            / "all_osdma8_biascorrected"
            / config.variable_setname
            / model_key
            / file_name
        )

    file_path.parent.mkdir(parents=True, exist_ok=True)

    return file_path