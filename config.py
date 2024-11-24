import lightgbm as lgb
from mlforecast.lag_transforms import (
    ExpandingMean,
    RollingMean,
    RollingQuantile,
    SeasonalRollingMax,
    SeasonalRollingMean,
    SeasonalRollingMin,
    SeasonalRollingStd,
)

config_project = {
    "catalog":"lingaro_sandbox_ne",
    "schema":"jf_sandbox",
}

config_model_params = {
    "lgb": {
        "verbose": -1,
        "num_leaves": 256,
        "n_estimators": 300,
        "objective": "tweedie",
        "tweedie_variance_power": 1.2809771485750954,
        "metric": "mae",
        "learning_rate": 0.057716139846986786,
        "lambda_l2": 0.5115166072323223,
        "lambda_l1": 1.3493074506435034,
    }
}

config = {
    "init": {
        "models": {
            "lgb": lgb.LGBMRegressor(**config_model_params["lgb"]),
        },
        "freq": "W-MON",
        "lags": [1, 13, 26, 52],
        "lag_transforms": {
            1: [
                RollingMean(window_size=52, min_samples=1),
                RollingMean(window_size=13, min_samples=1),
                RollingQuantile(window_size=52, p=0.5, min_samples=1),
                RollingQuantile(window_size=13, p=0.5, min_samples=1),
                ExpandingMean(),
                SeasonalRollingMean(52, 2, min_samples=1),
                SeasonalRollingStd(52, 2, min_samples=1),
                SeasonalRollingMax(52, 2, min_samples=1),
                SeasonalRollingMin(52, 2, min_samples=1),
            ],
            13: [
                RollingMean(window_size=13, min_samples=1),
                RollingQuantile(window_size=13, p=0.5, min_samples=1),
            ],
        },
        "target_transforms": [],
        "date_features": ["day", "month", "week", "year"],
    },
    "fit": {
        "static_features": ["Client", "Warehouse", "Product"],
        "target_col": "y",
        "time_col": "ds",
        "id_col": "unique_id",
        "h": 3,
        "n_windows": 4,
    },
    "reference_features": ["ds", "unique_id", "y"],
}
