import os

import pandas as pd
import xarray as xr

from fast_o3_iv.config import PROCESSED_DATA_DIR
from fast_o3_iv.preprocess.model_configs import MODEL_CONFIG


def has_expected_data(path, expected_timesteps):
    try:
        with xr.open_dataset(path, engine="netcdf4", decode_cf=False) as ds:
            if ds.sizes.get("time") != expected_timesteps:
                return False
            if len(ds.data_vars) == 0:
                return False
            return True
    except Exception:
        return False

def exists(path):
    return os.path.exists(path) and os.path.getsize(path) > 0

def _results_ds_dir(
    mode, source_id, experiment_id, variable_setname, model_key
):
    model_config = MODEL_CONFIG[source_id]
    institution_id = model_config["institution_id"]

    return (
        PROCESSED_DATA_DIR
        / "model_outputs"
        / "results"
        / mode
        / institution_id
        / source_id
        / experiment_id
        / "all"
        / variable_setname
        / model_key
    )


def _results_ds_path(
    mode, source_id, experiment_id, variable_setname, model_key
):
    model_config = MODEL_CONFIG[source_id]
    institution_id = model_config["institution_id"]

    out_dir = _results_ds_dir(
        mode, source_id, experiment_id, variable_setname, model_key
    )

    filename = (
        f"{institution_id}.{source_id}.{experiment_id}.all."
        f"{variable_setname}.{mode}.{model_key}.nc"
    )

    return out_dir / filename


def _parquet_file_to_ds(path):
    ensemble_member = path.parts[-4]
    print(ensemble_member, flush=True)

    df = pd.read_parquet(
        path,
        columns=["lat", "lon", "year", "month", "y_pred", "y_true"],
    ).copy()

    df["time"] = pd.to_datetime(
        df["year"] * 10000 + df["month"] * 100 + 1,
        format="%Y%m%d",
        cache=True,
    )

    ds = (
        df.drop(columns=["year", "month"])
        .set_index(["lat", "lon", "time"])
        .to_xarray()
        .expand_dims(ensemble_member=[ensemble_member])
    )

    return ds


def create_results_ds(
    mode,              # testing, inference, validation
    source_id,         # CESM2-WACCM6, CESM2
    experiment_id,     # historical, ssp245
    variable_setname,  # setA, setB
    model_key,         # GPR-local, MLR-local
):
    model_config = MODEL_CONFIG[source_id]
    institution_id = model_config["institution_id"]

    files = sorted(
        p for p in PROCESSED_DATA_DIR.glob(
            f"model_outputs/results/{mode}/{institution_id}/{source_id}/{experiment_id}/*/{variable_setname}/{model_key}/*"
        )
        if p.is_file()
    )

    print(f"Found {len(files)} parquet files")
    for f in files:
        print(f)

    datasets = [_parquet_file_to_ds(f) for f in files]

    ds = xr.concat(
        datasets,
        dim="ensemble_member",
        data_vars="minimal",
        coords="minimal",
        compat="override",
        join="exact",
        combine_attrs="override",
    )

    out_path = _results_ds_path(
        mode, source_id, experiment_id, variable_setname, model_key
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving to {out_path}")
    ds.to_netcdf(out_path)

    return ds

def load_or_create_results_ds(
    mode,              # testing, inference, validation
    source_id,         # CESM2-WACCM6, CESM2
    experiment_id,     # historical, ssp245
    variable_setname,  # setA, setB
    model_key,         # GPR-local, MLR-local
):
    model_config = MODEL_CONFIG[source_id]
    institution_id = model_config["institution_id"]

    existing_files = sorted(
        p for p in PROCESSED_DATA_DIR.glob(
            f"model_outputs/results/{mode}/{institution_id}/{source_id}/{experiment_id}/all/{variable_setname}/{model_key}/*"
        )
        if p.is_file()
    )

    if existing_files:
        out_path = existing_files[0]
        print(f"Loading existing dataset from {out_path}")
        return xr.open_dataset(out_path)

    print("No existing dataset found. Creating it now.")
    return create_results_ds(
        mode=mode,
        source_id=source_id,
        experiment_id=experiment_id,
        variable_setname=variable_setname,
        model_key=model_key,
    )
