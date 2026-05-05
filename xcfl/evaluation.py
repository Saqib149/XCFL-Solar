import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import List


def compute_metrics(y_true, y_pred, power_scale: float = 1.0) -> dict:
    """Return RMSE, MAE, and R² as a dict.  power_scale converts normalised
    [0,1] values to physical units (e.g. kW) so metrics match paper values."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred))) * power_scale
    mae  = float(mean_absolute_error(y_true, y_pred)) * power_scale
    r2   = float(r2_score(y_true, y_pred))
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def compute_per_client_metrics(clients, power_scale: float = 1.0) -> dict:
    """Average per-client RMSE / MAE / R² using each client's own model on its own test split."""
    rmse_list, mae_list, r2_list = [], [], []
    for client in clients:
        preds = client.predict(client.X_test)
        rmse_list.append(float(np.sqrt(mean_squared_error(client.y_test, preds))))
        mae_list.append(float(mean_absolute_error(client.y_test, preds)))
        r2_list.append(float(r2_score(client.y_test, preds)))
    return {
        "RMSE": float(np.mean(rmse_list)) * power_scale,
        "MAE":  float(np.mean(mae_list))  * power_scale,
        "R2":   float(np.mean(r2_list)),
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
    print(f"Metrics saved -> {path}")


def compare_methods(results: dict) -> pd.DataFrame:
    rows = [{"Method": name, **metrics} for name, metrics in results.items()]
    return pd.DataFrame(rows).set_index("Method")
