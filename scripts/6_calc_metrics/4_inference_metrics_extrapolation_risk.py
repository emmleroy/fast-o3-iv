import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from fast_o3_iv import utils
from fast_o3_iv.config import PROCESSED_DATA_DIR
#from fast_o3_iv.features.transforms import CyclicTransformer


parser = argparse.ArgumentParser()
parser.add_argument('-m', '--member', type=str, required=True)
args = parser.parse_args()


#############
N_JOBS = max(1, int(os.getenv("SLURM_CPUS_PER_TASK", "1")))
print(f"N_JOBS: {N_JOBS}", flush=True)
#############

# helper funcs
def gp_r_from_pipeline(pipeline, X_df):
    """take a fitted GPR pipeline and return the 
    GPR prediction (mean) and the extrapolation / uncertainty ratio (r)
    where r = posterior variance / prior variance

    r = 0 --> reduced uncertainty 
    r ~= 1 --> learned little relative to prior uncertainty (high extrapolation)
    """

    pre    = pipeline.named_steps["pre"]        # preprocessing
    scaler = pipeline.named_steps["scaler"]     # scaling
    gpr    = pipeline.named_steps["model"]      # fitted gpr model

    # if we have the input feature names, use them
    if hasattr(pre, "feature_names_in_"):
        X_df = X_df[pre.feature_names_in_]

    # use same scaling transformations
    X_scaled = scaler.transform(pre.transform(X_df))

    # grab mean and std
    mean, std = gpr.predict(X_scaled, return_std=True)

    # posterior variance = std²
    post_var_y = std**2

    # prior variance = k(x, x)
    prior_var_norm = gpr.kernel_.diag(X_scaled).astype(float)

    # undo normalize_y for the prior variance too
    ystd = getattr(gpr, "_y_train_std", 1.0)
    prior_var_y = prior_var_norm * (np.asarray(ystd) ** 2)

    # r should be between [0,1] --> clip for numerical effects
    r = post_var_y / np.maximum(prior_var_y, eps=1e-12)
    r = np.clip(r, 0.0, 1.0)

    # return the mean prediction and the extrapolation risk score
    return mean, r

def _compute_one_cell(idx, df_all, pipe):
    df_cell = df_all.loc[idx]
    mean, r = gp_r_from_pipeline(pipe, df_cell)
    return idx, mean, r

def add_gp_r_for_all_cells_parallel(
    df_all, model, n_jobs=-1, backend="threading", verbose=0
):
    """Compute GP mean and r for every row in df_all, parallel over grid cells.
    """
    mean_all = pd.Series(np.nan, index=df_all.index, dtype=float)
    r_all = pd.Series(np.nan, index=df_all.index, dtype=float)

    groups = df_all.groupby(["lat", "lon"]).groups

    tasks = []
    for (lat, lon), idx in groups.items():
        cell = (float(lat), float(lon))
        if cell in model:
            tasks.append((idx, model[cell]["pipeline"]))

    results = Parallel(
        n_jobs=n_jobs,
        backend=backend,
        verbose=verbose,
    )(
        delayed(_compute_one_cell)(idx, df_all, pipe)
        for idx, pipe in tasks
    )

    for idx, mean, r in results:
        mean_all.loc[idx] = mean
        r_all.loc[idx] = r

    df_out = df_all.copy()
    df_out["gp_mean"] = mean_all
    df_out["gp_r"] = r_all

    return df_out


## Main logic ##

# load prediction results
MODEL_KEY = "GPR-local"
df = utils.load_results(MODEL_KEY, "inference")

# load model object
path = PROCESSED_DATA_DIR / "models" / "NCAR.CESM2-WACCM6.wa6_ic1.001.historical.setL.GPR-local.joblib"
model = joblib.load(path) 

# add GPR extrapolation risk (`gpr_r`)
df_out = add_gp_r_for_all_cells_parallel(df, model, n_jobs=N_JOBS, backend="threading", verbose=10)

# save results
out_path = Path(
   PROCESSED_DATA_DIR / "model_outputs" / "inference" / f"NCAR.CESM2.{args.member}.historical.setL.{MODEL_KEY}.parquet"
)
out_path.parent.mkdir(parents=True, exist_ok=True)
df_out.to_parquet(out_path, index=False)
print(f"Saved: {out_path}", flush=True)
