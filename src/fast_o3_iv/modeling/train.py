from dataclasses import dataclass, asdict
import os
from typing import Sequence

import joblib
from joblib import Parallel, delayed
import mlflow
import numpy as np
import pandas as pd
from sklearn import set_config
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

from fast_o3_iv.config import PROCESSED_DATA_DIR
from fast_o3_iv.config import MLFLOW_TRACKING_URI, MODELS_DIR
from fast_o3_iv.features.transforms import CyclicTransformer
from fast_o3_iv.modeling.statmodel_registry import STATMODEL_REGISTRY
from fast_o3_iv.preprocess.model_configs import MODEL_CONFIG
from fast_o3_iv.tabularize.variable_setnames import VARIABLE_SET, pc_variable_sets

set_config(transform_output="pandas")


@dataclass 
class DatasetConfig:
    source_id: str             # 'CESM2-WACCM6'
    cmip_experiment_id: str    # historical or ssp245 -- note: avoid experiment_id used by MLflow
    variable_setname : str     # identifier for list of features to include
    year_range: Sequence[int]  # range(1975, 2015)

    def __post_init__(self): 
        self._model_config = MODEL_CONFIG[self.source_id]
        self.institution_id = self._model_config["institution_id"]
        self.variable_set = VARIABLE_SET[self.variable_setname]

        members_by_exp = self._model_config["members"] # used in predict.py
        self.members = members_by_exp[self.cmip_experiment_id]
        self.first_member = self.members[0] 
        self.second_member = self.members[1] if len(self.members) > 1 else []
        self.remaining_members = self.members[2:] if len(self.members) > 2 else []


class TrainPipeline:
    def __init__(self, model_key, config: DatasetConfig, is_test=False):
        
        statmodel_info = STATMODEL_REGISTRY[model_key]
        self.model_key = model_key
        self.stat_model = statmodel_info["model_obj"]
        self.scaler = statmodel_info["scaler"]

        self.config = config

        self.is_test = is_test

        # note: we hard-code the training-member as the first ensemble member
        self.training_member = self.config.first_member

        self.target = config.variable_set['target'][0]
        self.features = [
            sub_item
            for key, list_val in config.variable_set.items()
            if key != 'target'
            for item in list_val
            for sub_item in (pc_variable_sets[item] if item in pc_variable_sets else [item])
        ]

        # check that variable sets are correct
        if "global" in self.model_key:
            if "location_variables" not in config.variable_set.keys():
                raise ValueError("for global training, variable_set should have location vars")
        elif "local" in self.model_key:
            if "location_variables" in config.variable_set.keys():
                raise ValueError("for local training, variable_set does not need location vars")

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        experiment = mlflow.get_experiment_by_name("fast-o3-iv")
        if experiment is None:
            mlflow.create_experiment(
                name="fast-o3-iv",
            )
        mlflow.set_experiment("fast-o3-iv")

    def _get_df_filepath(self, member):

        file_name = (
            f"{self.config.institution_id}."
            f"{self.config.source_id}."
            f"{member}."
            f"{self.config.cmip_experiment_id}."
            f"{self.config.variable_setname}."
            "inputs.parquet"
        )
        file_path = (
            PROCESSED_DATA_DIR
            / "model_inputs"
            / file_name
        )
        return file_path

    def _get_model_filepath(self):
        if self.is_test:
            suffix = ".TEST"
        else:
            suffix = ""

        file_name = (
            f"{self.config.institution_id}."
            f"{self.config.source_id}."
            f"{self.training_member}."
            f"{self.config.cmip_experiment_id}."
            f"{self.config.variable_setname}."
            f"{self.model_key}{suffix}.joblib"
        )
        file_path = MODELS_DIR / file_name

        return file_path

    def _load_df(self, member):
        file_path = self._get_df_filepath(member)
        df = pd.read_parquet(file_path)
        return df

    def fit_globally(self, save=True):
        """Fit a single model across all grid cells
        """
        run_name = self.model_key + '-' + self.config.cmip_experiment_id
        with mlflow.start_run(run_name=run_name):
            member = self.training_member
            df = self._load_df(member)

            if self.is_test:
                print("Testing on a small subset of df", flush=True)
                df_test = df.copy()
                keys = (
                    df_test[["lat", "lon"]]
                    .drop_duplicates()
                    .sample(n=4, random_state=42)
                )
                df = df_test.merge(keys, on=["lat", "lon"], how="inner")

            X = df.loc[:, self.features]
            y = df.loc[:, self.target].to_numpy()

            # pre-processing steps
            pre = ColumnTransformer(
                transformers=[
                ("month_cyc", CyclicTransformer("month", period=12, drop_original=True), ["month"]),
                ("lon_cyc",   CyclicTransformer("lon",   period=360, drop_original=True), ["lon"]),
            ],
                remainder="passthrough",
                verbose_feature_names_out=False,  
            ).set_output(transform="pandas")

            # define pipeline
            pipeline = Pipeline([
                ("pre",    pre),
                ("scaler", clone(self.scaler)),
                ("model",  clone(self.stat_model)),
            ]).set_output(transform="pandas")

            # fit the model
            pipeline.fit(X, y)

            # training metrics
            y_pred = pipeline.predict(X)
            rmse = float(np.sqrt(mean_squared_error(y, y_pred)))

            ########### mlflow logging ###########
            # 1. input data configs
            mlflow.log_params(asdict(self.config))

            # 2. model type
            mlflow.log_params({
                "model_type": self.stat_model.__class__.__name__,
            })

            # 3. is_test
            mlflow.set_tag("is_test", str(self.is_test))

            # 4. model path (manual save instead of using mlflow model logging - maybe change this?)
            if save:
                final_path = self._get_model_filepath()
                mlflow.log_param("local_disk_path", str(final_path))

            # 5. save training metrics
            # note! these are NOT our selection metrics (area-weighted mean in populated grid cells
            # after per-cell predictions) -- we just internally track this to compare within-model 
            # diagnostics (train vs val), not for model comparison/model selection (e.g., global vs local)
            mlflow.log_metric("train_rmse_global_unweighted", float(rmse))
            #######################################

            return {
                "pipeline": pipeline,
            }

    def fit_locally(self, save=True):
        """Fit a unique model per grid cell in parallel, 
        then log the collection as one artifact in MLflow.
        """

        run_name = self.model_key + '-' + self.config.cmip_experiment_id
        with mlflow.start_run(run_name=run_name):
            member = self.training_member
            df = self._load_df(member)
            
            if self.is_test:
                print("Testing on a small subset of df", flush=True)
                df_test = df.copy()
                keys = (
                    df_test[["lat", "lon"]]
                    .drop_duplicates()
                    .sample(n=4, random_state=42)
                )
                df = df_test.merge(keys, on=["lat", "lon"], how="inner")

            # group by lat/lon
            groups = df.groupby(["lat", "lon"], sort=False)

            # use Joblib to fit in parallel
            n_cpus = int(os.getenv("SLURM_CPUS_PER_TASK", str(os.cpu_count() or 1)))
            print(f"Parallelizing fit over {len(groups)} cells using {n_cpus} CPUs...")
            results = Parallel(n_jobs=n_cpus)(
                delayed(self._fit_single_cell_with_progress)(key, cell_df)
                for key, cell_df in groups
            )

            # map results back to (lat, lon) keys
            cell_models = {key: pipe for key, pipe, _ in results}

            # aggregate RMSE only for logging
            rmses = np.array([rmse for _, _, rmse in results], dtype=float)

            ########### mlflow logging ###########
            # 1. input data configs
            mlflow.log_params(asdict(self.config))

            # 2. model type
            mlflow.log_params({
                "model_type": self.stat_model.__class__.__name__,
            })

            # 3. is_test
            mlflow.set_tag("is_test", str(self.is_test))

            # 4. model path (manual save instead of using mlflow model logging - maybe change this?)
            if save:
                final_path = self._get_model_filepath()
                mlflow.log_param("local_disk_path", str(final_path))

            # 5. save training metrics
            # note! these are NOT our selection metrics (area-weighted mean in populated grid cells
            # after per-cell predictions) -- we just internally track this to compare within-model 
            # diagnostics (train vs val), not for model comparison/model selection (e.g., global vs local)
            mlflow.log_metric("train_rmse_local_mean_unweighted", float(rmses.mean()))
            mlflow.log_metric("train_rmse_local_median_unweighted", float(np.median(rmses)))
            #######################################

            return {
                "pipeline_collection": cell_models,
            }

    def _fit_single_cell(self, cell_df):
        X = cell_df.loc[:, self.features]
        y = cell_df.loc[:, self.target].to_numpy()

        # pre-processing steps
        pre = ColumnTransformer(
            transformers=[
            ("month_cyc", CyclicTransformer("month", period=12, drop_original=True), ["month"]),
        ],
            remainder="passthrough",
            verbose_feature_names_out=False,  
        ).set_output(transform="pandas")

        pipeline = Pipeline([
            ("pre", pre),
            ("scaler", clone(self.scaler)),
            ("model", clone(self.stat_model)) # each cell gets own copy of the model
        ]).set_output(transform="pandas")

        pipeline.fit(X, y)

        y_pred = pipeline.predict(X)
        rmse = float(np.sqrt(mean_squared_error(y, y_pred)))

        return pipeline, rmse

    def _fit_single_cell_with_progress(self, key, cell_df):
        pipe, rmse = self._fit_single_cell(cell_df)
        print(f"Completed group {key}", flush=True)
        return key, pipe, rmse

    def train_and_save_model(self, save=True):
        out_path = self._get_model_filepath()
        if save:
            if out_path.exists():
                raise FileExistsError(f"Refusing to overwrite existing file: {out_path}")

        core_model = None

        if "global" in self.model_key:
            core_model = self.fit_globally(save=save)
            save_obj = core_model["pipeline"]

        elif "local" in self.model_key:
            core_model = self.fit_locally(save=save)
            save_obj = core_model["pipeline_collection"]
        
        else:
            raise ValueError("model_key must include 'global' or 'local' ")
        print(f"Ready to save model --> {self.model_key} --> {out_path}", flush=True)

        if save and save_obj is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(save_obj, out_path)
            print(f"Model saved --> {self.model_key}", flush=True)

