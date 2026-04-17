import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """Return RMSE, MAE, and R² as a dict."""
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def print_metrics(metrics: dict, label: str = "Results") -> None:
    print(f"\n===== {label} =====")
    for key, val in metrics.items():
        print(f"  {key:<6}: {val:.4f}")


def save_metrics(metrics: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pd.DataFrame({"Metric": list(metrics.keys()), "Value": list(metrics.values())}).to_csv(
        path, index=False
    )
    print(f"Metrics saved → {path}")


def compare_methods(results: dict) -> pd.DataFrame:
    """
    Build a comparison table from a dict of {method_name: metrics_dict}.

    Example
    -------
    compare_methods({
        "XCFL":        {"RMSE": 0.10, "MAE": 0.05, "R2": 0.68},
        "FedAvg":      {"RMSE": 0.12, "MAE": 0.06, "R2": 0.63},
        "Centralized": {"RMSE": 0.09, "MAE": 0.04, "R2": 0.71},
    })
    """
    rows = [{"Method": name, **metrics} for name, metrics in results.items()]
    df = pd.DataFrame(rows).set_index("Method")
    return df
