import os
import joblib
from joblib import Parallel, delayed
from typing import Sequence
from dataclasses import dataclass, asdict

import mlflow
import pandas as pd

from sklearn import set_config
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler # scaling method
from sklearn.base import clone

from fast_o3_iv.config import PROCESSED_DATA_DIR
from fast_o3_iv.config import MLFLOW_TRACKING_URI, MODELS_DIR
from fast_o3_iv.preprocess.model_configs import MODEL_CONFIG
from fast_o3_iv.tabularize.variable_setnames import VARIABLE_SET, pc_variable_sets
from fast_o3_iv.features.transforms import CyclicTransformer
from fast_o3_iv.modeling.statmodel_registry import STATMODEL_REGISTRY

set_config(transform_output="pandas")


@dataclass 
class DatasetConfig:
    source_id: str             # 'CESM2-WACCM6'
    cmip_experiment_id: str    # historical or ssp245 -- note: avoid experiment_id used by MLflow
    variable_setname : str     # identifier for list of features to include
    training_member: str       # training ensemble member ID
    year_range: Sequence[int]  # range(1975, 2015)

    def __post_init__(self): 
        self._model_config = MODEL_CONFIG[self.source_id]
        self.institution_id = self._model_config["institution_id"]
        self.variable_set = VARIABLE_SET[self.variable_setname]


class TrainPipeline:
    def __init__(self, model_key, run_name, config: DatasetConfig, is_test=False):
        
        statmodel_info = STATMODEL_REGISTRY[model_key]
        self.model_key = model_key
        self.stat_model = clone(statmodel_info["model_obj"])

        self.run_name = run_name
        self.config = config

        self.is_test = is_test

        # derived attrs
        self.target = config.variable_set['target'][0]
        self.features = [
            sub_item
            for key, list_val in config.variable_set.items()
            if key != 'target'
            for item in list_val
            for sub_item in (pc_variable_sets[item] if item in pc_variable_sets else [item])
        ]

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        experiment = mlflow.get_experiment_by_name("fast-o3-iv")
        if experiment is None:
            mlflow.create_experiment(
                name="fast-o3-iv",
            )
        mlflow.set_experiment("fast-o3-iv")

    def _get_df_filepath(self):
        file_name = (
            f"{self.config.institution_id}."
            f"{self.config.source_id}."
            f"{self.config.cmip_experiment_id}."
            f"{self.config.training_member}."
            f"{self.config.variable_setname}."
            "model_inputs."
            f"{self.config.year_range[0]}-{self.config.year_range[-1]}.parquet"
        )
        file_path = (
            PROCESSED_DATA_DIR
            / "model_inputs"
            / self.config.institution_id
            / self.config.source_id
            / self.config.cmip_experiment_id
            / self.config.training_member
            / self.config.variable_setname
            / file_name
        )
        return file_path 

    def _get_model_filepath(self):
        file_name = (
            f"{self.config.institution_id}."
            f"{self.config.source_id}."
            f"{self.config.cmip_experiment_id}."
            f"{self.config.training_member}."
            f"{self.config.variable_setname}."
            "model."
            f"{self.config.year_range[0]}-{self.config.year_range[-1]}."
            f"{self.run_name}.joblib"
        )
        file_path = (
            MODELS_DIR
            / self.config.institution_id
            / self.config.source_id
            / self.config.cmip_experiment_id
            / self.config.training_member
            / self.config.variable_setname
            / file_name
        )

        return file_path

    def load_training_df(self):
        file_path = self._get_df_filepath()
        df = pd.read_parquet(file_path)
        return df

    def fit_globally(self, save=True):
        """Fit a single model across all grid cells
        """
        with mlflow.start_run(run_name=self.run_name):
            df = self.load_training_df()

            if self.is_test:
                print("Testing on a small subset of df", flush=True)
                df_test = df.copy()
                df = df_test[ (df_test['lat'] - 20.0).abs() < 1 ] # test on subset

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
                ("scaler", RobustScaler()),
                ("model",  clone(self.stat_model)),
            ]).set_output(transform="pandas")

            # fit the model
            pipeline.fit(X, y)

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
            #######################################

            return {
                "pipeline": pipeline,
            }

    def fit_locally(self, save=True):
        """Fit a unique model per grid cell in parallel, 
        then log the collection as one artifact in MLflow.
        """
        with mlflow.start_run(run_name=self.run_name):
            df = self.load_training_df()
            
            if self.is_test:
                print("Testing on a small subset of df", flush=True)
                df_test = df.copy()
                df = df_test[ (df_test['lat'] - 20.0).abs() < 1 ] # test on subset

            # group by lat/lon
            groups = df.groupby(["lat", "lon"], sort=False)
            print(f"Parallelizing fit over {len(groups)} cells using {os.cpu_count()} CPUs...")

            # use Joblib to fit in parallel
            n_cpus = int(os.getenv("SLURM_CPUS_PER_TASK", "1"))
            cell_models_list = Parallel(n_jobs=n_cpus)( # gdf = group dataframe
                delayed(self._fit_single_cell_with_progress)(key, gdf) 
                for key, gdf in groups
            )

            # map results back to (lat, lon) keys
            cell_models = dict(zip(groups.groups.keys(), cell_models_list))

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
            #######################################

            return {
                "pipeline_collection": cell_models,
            }

    def _fit_single_cell(self, cell_df):
        """scikit-learn logic for one cell"""
        X = cell_df.loc[:, self.features]
        y = cell_df.loc[:, self.target].to_numpy()

        # Month cyclic only (as per your local script)
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
            ("scaler", RobustScaler()),
            ("model", clone(self.stat_model)) # each cell gets own copy of the model
        ]).set_output(transform="pandas")

        return pipeline.fit(X, y)

    def _fit_single_cell_with_progress(self, key, gdf):
        """Wrapper to call fit and print progress for a specific lat/lon"""
        result = self._fit_single_cell(gdf)
        # Flush=True is crucial for Slurm logs to update in real-time
        print(f"Completed group {key}", flush=True)
        return result

    def train_and_save_model(self, save=True):
        out_path = self._get_model_filepath()
        core_model = None

        if "global" in self.model_key:
            core_model = self.fit_globally(save=save)
            save_obj = core_model["pipeline"]

        elif "local" in self.model_key:
            core_model = self.fit_locally(save=save)
            save_obj = core_model["pipeline_collection"]
        
        else:
            raise AttributeError("model_key must include 'global' or 'local' ")
        print(f"Ready to save model --> {self.run_name} --> {out_path}", flush=True)

        if save and save_obj is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(save_obj, out_path)
            print(f"Model saved --> {self.run_name}", flush=True)

