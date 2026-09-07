
import pandas as pd
import numpy as np 

from fast_o3_iv import metrics, utils
from fast_o3_iv.config import PROCESSED_DATA_DIR


MODEL_KEYS = [
    "MLR-global",
    "XGB-global",
    "GPR-local",
    "MLR-local",
    "XGB-local",
]

TEST_METRICS = [
    "ens_clim", 
    "ens_trend", 
    "resid_sd", 
    "resid_q90",
    "pgt_q90", 
    "emd_to_noise"
]

results = {}

for model_key in MODEL_KEYS:
    df = utils.load_results(model_key, "testing")

    summary = metrics.calculate_global_test_metrics(df, only_populated_grid_cells=True)

    results[model_key] = {
        metric: np.round(float(summary[metric]), 2) for metric in TEST_METRICS
    }

table = pd.DataFrame(results).reindex(TEST_METRICS)

print(table)

table.to_csv(PROCESSED_DATA_DIR / "figure_data" / "Table2_test_metrics.csv")
