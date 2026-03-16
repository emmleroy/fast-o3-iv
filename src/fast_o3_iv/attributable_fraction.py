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


def calc_af_range(osdma8_ds, osdma8_var):
    """
    Calculate central and sensitivity AF estimates for one OSDMA8 variable.

    Returns:
    - central_af: central GBD-style AF
    - low_af: lower-burden bound (low RR, high TMREL)
    - high_af: upper-burden bound (high RR, low TMREL)
    - rr_low_af / rr_high_af: RR-only sensitivity
    - tmrel_low_af / tmrel_high_af: TMREL-only sensitivity
    """
    if osdma8_var not in osdma8_ds:
        raise KeyError(f"{osdma8_var!r} not found in dataset")

    da = osdma8_ds[osdma8_var]
    out = xr.Dataset()

    def calc_af_mini(da, tmrel=32.4, rr_per_10ppb=1.063):
        beta = np.log(rr_per_10ppb) / 10.0
        excess = xr.where(da > tmrel, da - tmrel, 0.0)
        rr = np.exp(beta * excess)
        af = (rr - 1.0) / rr * 100.0
        return af


    # Central estimate
    out[f"{osdma8_var}_central_af"] = calc_af_mini(
        da,
        tmrel=32.4,
        rr_per_10ppb=1.063,
    )

    # Combined low- and high-burden bounds
    out[f"{osdma8_var}_low_af"] = calc_af_mini(
        da,
        tmrel=35.7,       # higher TMREL -> lower burden
        rr_per_10ppb=1.029,
    )

    out[f"{osdma8_var}_high_af"] = calc_af_mini(
        da,
        tmrel=29.1,       # lower TMREL -> higher burden
        rr_per_10ppb=1.098,
    )

    # RR-only sensitivity
    out[f"{osdma8_var}_rr_low_af"] = calc_af_mini(
        da,
        tmrel=32.4,
        rr_per_10ppb=1.029,
    )

    out[f"{osdma8_var}_rr_high_af"] = calc_af_mini(
        da,
        tmrel=32.4,
        rr_per_10ppb=1.098,
    )

    # TMREL-only sensitivity
    out[f"{osdma8_var}_tmrel_low_af"] = calc_af_mini(
        da,
        tmrel=29.1,
        rr_per_10ppb=1.063,
    )

    out[f"{osdma8_var}_tmrel_high_af"] = calc_af_mini(
        da,
        tmrel=35.7,
        rr_per_10ppb=1.063,
    )

    if "lat" in out.coords:
        out = out.sel(lat=slice(-60, 75))

    out.attrs.update({
        "af": "Attributable fraction (%) of COPD mortality attributable to ozone",
        "units": "percent",
        "health_relevant_ozone_metric": "OSDMA8",
        "methodology": "GBD 2019-style COPD AF sensitivity calculation",
        "central_tmrel_ppbv": 32.4,
        "central_rr_per_10ppbv": 1.063,
        "tmrel_range_ppbv": "29.1 to 35.7",
        "rr_range_per_10ppbv": "1.029 to 1.098",
        "note": (
            "low_af and high_af are bounding sensitivity cases, not a full "
            "probabilistic uncertainty interval"
        ),
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