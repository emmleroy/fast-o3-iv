from dataclasses import asdict
import os

import joblib
from joblib import Parallel, delayed
import mlflow
import pandas as pd
import numpy as np
from sklearn import set_config
from sklearn.metrics import mean_squared_error

from fast_o3_iv.config import PROCESSED_DATA_DIR
from fast_o3_iv.modeling.train import TrainPipeline, DatasetConfig

set_config(transform_output="pandas")


class PredictPipeline(TrainPipeline):
    def __init__(self, model_key, train_config: DatasetConfig, predict_config: DatasetConfig, is_test=False):
        super().__init__(model_key=model_key, config=train_config, is_test=is_test)
        self.train_config = train_config
        self.predict_config = predict_config
        self.validation_member = self.train_config.second_member

    def load_model(self):
        """Load the model corresponding to model_key and config

        Note this is pretty slow for GPR-local (takes about ~5 mins)"""
        model_path = self._get_model_filepath()  # uses training config via inheritance
        model_object = joblib.load(model_path)

        return model_object

    def _get_df_filepath(self, member):
        """Override inherited _get_df_filepath from TrainPipeline with 
        cfg = self.predict_config 

        note: this is for input dfs
        """
        cfg = self.predict_config
        file_name = (
            f"{cfg.institution_id}."
            f"{cfg.source_id}."
            f"{cfg.cmip_experiment_id}."
            f"{member}."
            f"{cfg.variable_setname}."
            "model_inputs."
            f"{cfg.year_range[0]}-{cfg.year_range[-1]}.parquet"
        )
        file_path = (
            PROCESSED_DATA_DIR
            / "model_inputs"
            / cfg.institution_id
            / cfg.source_id
            / cfg.cmip_experiment_id
            / member
            / cfg.variable_setname
            / file_name
        )
        return file_path

    def _get_output_df_filepath(self, member):
        """Filepath to save model_outputs to
        """
        #if self.is_test:
        #    suffix = ".TEST"
        #else:
        #    suffix = ""

        #train_cfg = self.train_config
        pred_cfg = self.predict_config
        
        file_name = (
            f"{pred_cfg.institution_id}."
            f"{pred_cfg.source_id}."
            f"{pred_cfg.cmip_experiment_id}."
            f"{member}."
            f"{pred_cfg.variable_setname}."
            f"{pred_cfg.year_range[0]}-{pred_cfg.year_range[-1]}."
            f"{self.model_key}.parquet"
            #f"trained-on-{train_cfg.source_id}-{train_cfg.cmip_experiment_id}{suffix}.parquet" # option if we start mixing train/predict scenarios or models
        )
        file_path = (
            PROCESSED_DATA_DIR
            / "model_outputs"
            / "member_results"
            / pred_cfg.institution_id
            / pred_cfg.source_id
            / pred_cfg.cmip_experiment_id
            / member
            / pred_cfg.variable_setname
            / f"{self.model_key}"
            / file_name
        )
        return file_path

    def predict_globally(self, member, df, pipeline):
       
        if "global" not in self.model_key:
            raise NotImplementedError("predict_globally() should use a global model key")

        run_name = f"{self.model_key}-{self.predict_config.cmip_experiment_id}-{self.mode}"
        with mlflow.start_run(run_name=run_name):
            
            if self.is_test:
                print(f"{self.mode} on a small subset of df", flush=True)
                df_test = df.copy()
                keys = (
                    df_test[["lat", "lon"]]
                    .drop_duplicates()
                    .sample(n=4, random_state=42)
                )
                df = df_test.merge(keys, on=["lat", "lon"], how="inner")
            
            X = df.loc[:, self.features]

            y_pred = pipeline.predict(X)

            y = None
            rmse = None
            # val / test metrics
            if self.mode in ['validation', 'testing']:
                y = df.loc[:, self.target].to_numpy()
                rmse = float(np.sqrt(mean_squared_error(y, y_pred)))

            ########### mlflow logging ###########
            # 1. input data configs
            train_params = asdict(self.train_config)
            predict_params = asdict(self.predict_config)

            mlflow.log_params({f"train_{k}": v for k, v in train_params.items()})
            mlflow.log_params({f"predict_{k}": v for k, v in predict_params.items()})

            # 2. model type & valdation member
            mlflow.log_params({
                "model_type": self.stat_model.__class__.__name__,
                "prediction_member": member,
                "prediction_mode": self.mode,
            })

            # 3. is_test
            mlflow.set_tag("is_test", str(self.is_test))

            # 5. save training metrics
            # note! these are NOT our selection metrics (area-weighted mean in populated grid cells
            # after per-cell predictions) -- we just internally track this to compare within-model
            # diagnostics (train vs val), not for model comparison/model selection (e.g., global vs local)
            if rmse is not None:
                mlflow.log_metric(f"{self.mode}_rmse_global_unweighted", float(rmse))
            #######################################

            # create results dataframe
            cols_keep = [c for c in ["lat", "lon", "year", "month", "member"] if c in df.columns]
            results_df = df[cols_keep].copy()
            results_df["y_pred"] = y_pred

            if y is not None:
                results_df["y_true"] = y

            return results_df

    def predict_locally(self, member, df, pipeline):

        if "local" not in self.model_key:
            raise NotImplementedError("predict_locally() should use a local model key")

        run_name = f"{self.model_key}-{self.predict_config.cmip_experiment_id}-{self.mode}"
        with mlflow.start_run(run_name=run_name):
            
            if self.is_test:
                print(f"{self.mode} on a small subset of df", flush=True)
                df_test = df.copy()
                keys = (
                    df_test[["lat", "lon"]]
                    .drop_duplicates()
                    .sample(n=4, random_state=42)
                )
                df = df_test.merge(keys, on=["lat", "lon"], how="inner")
            
            groups = df.groupby(["lat", "lon"], sort=False)

            tasks = [
                (key, group.copy(deep=False), pipeline[key]) for key, group in groups
            ]

            # use Joblib to predict in parallel
            n_cpus = int(os.getenv("SLURM_CPUS_PER_TASK", str(os.cpu_count() or 1)))
            print(f"Parallelizing prediction over {len(groups)} cells using {n_cpus} CPUs...")
            results = Parallel(n_jobs=n_cpus, backend="threading", verbose=5)(
                delayed(self._predict_single_cell)(key, group, model) for key, group, model in tasks
            )

            ########### mlflow logging ###########
            # 1. input data configs
            train_params = asdict(self.train_config)
            predict_params = asdict(self.predict_config)

            mlflow.log_params({f"train_{k}": v for k, v in train_params.items()})
            mlflow.log_params({f"predict_{k}": v for k, v in predict_params.items()})

            # 2. model type & valdation member
            mlflow.log_params({
                "model_type": self.stat_model.__class__.__name__,
                "prediction_member": member,
                "prediction_mode": self.mode,
            })

            # 3. is_test
            mlflow.set_tag("is_test", str(self.is_test))

            # 4. combine results datataframes (df_pred) from predict_one_cell()
            pred_frames = [r["predictions"] for r in results]
            results_df = pd.concat(pred_frames, ignore_index=True)

            # 5. save training metrics
            # note! these are NOT our selection metrics (area-weighted mean in populated grid cells
            # after per-cell predictions) -- we just internally track this to compare within-model
            # diagnostics (train vs val), not for model comparison/model selection (e.g., global vs local)
            rmse = None
            # val / test metrics
            if self.mode in ['validation', 'testing']:
                y_true = results_df['y_true']
                y_pred = results_df['y_pred']
                rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

            if rmse is not None:
                mlflow.log_metric(f"{self.mode}_rmse_global_unweighted", float(rmse))
            #######################################

            return results_df

    def _predict_single_cell(self, key, cell_df, cell_pipeline):
        """Apply model.predict() for each cell
        """

        # define X and y_true
        X = cell_df.loc[:, self.features]
        y_pred = cell_pipeline.predict(X)

        y = None
        if self.mode in ['validation', 'testing']:
            y = cell_df.loc[:, self.target].to_numpy()

        # create results dataframe
        cols_keep = [c for c in ["lat", "lon", "year", "month", "member"] if c in cell_df.columns]
        cell_results_df = cell_df[cols_keep].copy()
        cell_results_df["y_pred"] = y_pred

        if y is not None:
            cell_results_df["y_true"] = y

        return {"predictions": cell_results_df}


    def predict_and_save_results(self, member, pipeline, save=True):

        df = self._load_df(member)
        has_target = self.target in df.columns

        if has_target:
            if member == self.validation_member:
                self.mode = 'validation'
            else:
                self.mode = 'testing'
        else:
            self.mode = 'inference'

        out_path = self._get_output_df_filepath(member)

        if save and out_path.exists():
            print(f"Skipping existing file: {out_path}", flush=True)
            return

        if "global" in self.model_key:
            results = self.predict_globally(member, df, pipeline)
            
        elif "local" in self.model_key:
            results = self.predict_locally(member, df, pipeline)
        
        else:
            raise ValueError("model_key must include 'global' or 'local' ")
        print(f"Ready to save results --> {self.model_key} --> {out_path}", flush=True)

        if save and results is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            results.to_parquet(out_path, index=False)
            print(f"Saved: {out_path}", flush=True)
