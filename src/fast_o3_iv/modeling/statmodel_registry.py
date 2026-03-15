# model_registry.py
#from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler 
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel as C,
    Matern,
)
from scipy.stats import uniform, loguniform
import numpy as np

#########################################
# GPR Kernels
def two_scale_kernel(num_features_post_transform):
    """Helper func to build the k_two_scale kernel based on feature count.
    """
    amp = (1e-3, 1e3)
    ls = np.ones(num_features_post_transform)
    kernel = (
        C(1.0, amp) * Matern(1.0 * ls, (1e-1, 1e3), nu=2.5)
        + C(1.0, amp) * Matern(0.2 * ls, (1e-3, 1e1), nu=0.5)
    )
    return kernel

def single_scale_kernel(num_features_post_transform):
    ls = np.ones(num_features_post_transform)
    kernel = (
        # nonlinear part with its own amplitude
        C(1.0, (1e-3, 1e3))
        * Matern(
            length_scale=ls,
            length_scale_bounds=(1e-2, 1e2),
            nu=2.5,
        )
    )
    return kernel
#########################################


STATMODEL_REGISTRY = {
    "MLR-global": {
        "model_obj": LinearRegression(),
        "scaler": RobustScaler(),
        "param_grid": None,
    },
    "MLR-local": {
        "model_obj": LinearRegression(),
        "scaler": RobustScaler(),
        "param_grid": None,
    },
    #"XGB-global": {
    #    "model_obj": XGBRegressor(
    #        objective="reg:squarederror",
    #        random_state=42,
    #        tree_method="hist",
    #    ),
    #    "scaler": RobustScaler(),
    #    "param_grid": {
    #        "model__max_depth": range(2, 12, 2),
    #        "model__learning_rate": np.arange(0.1, 0.3, 0.03),
    #        "model__subsample": uniform(0.6, 0.4),
    #        "model__colsample_bytree": uniform(0.6, 0.4),
    #        "model__gamma": loguniform(1e-8, 10.0),
    #        "model__reg_alpha": loguniform(1e-8, 100.0),
    #        "model__reg_lambda": loguniform(1e-2, 100.0),
    #        "model__min_child_weight": [1, 3, 5, 10, 20],
    #    },
    #},
    #"XGB-local" : {
    #    "model_obj": XGBRegressor(
    #        objective="reg:squarederror",
    #        random_state=42,
    #        tree_method="hist",
    #    ),
    #    "scaler": RobustScaler(),
    #    "param_grid": {
    #        "model__max_depth": range(2, 12),
    #        "model__learning_rate": loguniform(0.03, 0.3),
    #        "model__subsample": uniform(0.6, 0.4),
    #        "model__colsample_bytree": uniform(0.6, 0.4),
    #        "model__gamma": loguniform(1e-8, 5.0),
    #        "model__reg_alpha": loguniform(1e-8, 10.0),
    #        "model__reg_lambda": loguniform(0.1, 10.0),
    #        "model__min_child_weight": [1, 3, 5, 10, 20],
    #    },
    #},
    "GPR-local": {
        "model_obj": GaussianProcessRegressor(
            kernel=two_scale_kernel(num_features_post_transform=19), # note!! hard coded 18+1
            normalize_y=True,
            n_restarts_optimizer=0,
            random_state=42,
            alpha=1e-10,
        ),
        "scaler": RobustScaler(),
        "param_grid": None,
    },
    "GPR-local-two-standard": {
        "model_obj": GaussianProcessRegressor(
            kernel=two_scale_kernel(num_features_post_transform=19), # note!! hard coded 18+1
            normalize_y=True,
            n_restarts_optimizer=0,
            random_state=42,
            alpha=1e-10,
        ),
        "scaler": StandardScaler(with_mean=True, with_std=True),
        "param_grid": None,
    },
    "GPR-local-single-robust": {
        "model_obj": GaussianProcessRegressor(
            kernel=single_scale_kernel(num_features_post_transform=19), # note!! hard coded 18+1
            normalize_y=True,
            n_restarts_optimizer=2,
            random_state=42,
            alpha=1e-10,
        ),
        "scaler": RobustScaler(),
        "param_grid": None,
    },
    "GPR-local-single-standard": {
        "model_obj": GaussianProcessRegressor(
            kernel=single_scale_kernel(num_features_post_transform=19), # note!! hard coded 18+1
            normalize_y=True,
            n_restarts_optimizer=2,
            random_state=42,
            alpha=1e-10,
        ),
        "scaler": StandardScaler(with_mean=True, with_std=True),
        "param_grid": None,
    },
}