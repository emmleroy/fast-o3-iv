from dataclasses import dataclass, asdict

import joblib
from pathlib import Path
from typing import Sequence

import mlflow
import numpy as np
import pandas as pd

from sklearn import set_config
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler # scaling method

from fast_o3_iv.config import PROJ_ROOT, PROCESSED_DATA_DIR
from fast_o3_iv.config import MLFLOW_TRACKING_DIR, MLFLOW_TRACKING_URI
from fast_o3_iv.preprocess.model_configs import MODEL_CONFIG
from fast_o3_iv.tabularize.variable_setnames import VARIABLE_SET, pc_variable_sets
from fast_o3_iv.features.transforms import CyclicTransformer

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
    def __init__(self, stat_model, run_name, config: DatasetConfig):
        self.stat_model = stat_model
        self.run_name = run_name
        self.config = config

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
                artifact_location=MLFLOW_TRACKING_DIR
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
            PROJ_ROOT
            / "models"
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

    def fit_globally(self, is_test=False):
        """Fit a single model across all grid cells
        """

        with mlflow.start_run(run_name=self.run_name) as run:
            run_id = run.info.run_id

            # save the config inputs
            mlflow.log_params(asdict(self.config))
            mlflow.log_params({
                "is_test" : is_test,
                "model_type": self.stat_model.__class__.__name__,
            })

            df = self.load_training_df()

            if is_test:
                print("Testing on a small subset of df", flush=True)
                df_test = df.copy()
                df = df_test[ (df_test['lat'] - 20.0).abs() < 1 ] # test on subset
            else:
                df = df.copy()

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
                ("model",  self.stat_model),
            ]).set_output(transform="pandas")

            # fit the model
            pipeline.fit(X, y)

            # log model
            mlflow.sklearn.log_model(sk_model=pipeline, name="model")

            return {
                "pipeline": pipeline,
                "mlflow_run_id": run_id
            }
    
    def train_and_save_model(self, is_test=False, save=True):
        out_path = self._get_model_filepath()
        results = None

        if "global" in self.run_name:
            results = self.fit_globally(is_test=is_test)

        print(f"Ready to save model --> {self.run_name} --> {out_path}")

        if save and results is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(results, out_path)

