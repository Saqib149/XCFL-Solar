import os
import pandas as pd
from dataclasses import dataclass, field
from typing import List


def select_features_by_pearson(
    data_folder: str,
    target_col: str = "power_normed",
    n_features: int = 8,
    delimiter: str = ";",
) -> List[str]:
    """
    Load all client CSVs, compute Pearson correlation of every numeric feature
    with target_col, and return the top n_features by absolute correlation value.
    """
    csv_files = sorted(f for f in os.listdir(data_folder) if f.endswith(".csv"))
    frames = [
        pd.read_csv(os.path.join(data_folder, f), delimiter=delimiter)
        for f in csv_files
    ]
    combined = pd.concat(frames, ignore_index=True)

    numeric_cols = combined.select_dtypes(include="number").columns.tolist()
    feature_candidates = [c for c in numeric_cols if c != target_col]

    corr = combined[feature_candidates].corrwith(combined[target_col]).abs()
    top_features = corr.nlargest(n_features).index.tolist()

    print(f"\n[Pearson] Top {n_features} features selected (|correlation| with '{target_col}'):")
    for feat in top_features:
        print(f"  {feat:40s}  r = {corr[feat]:.4f}")

    return top_features


@dataclass
class XCFLConfig:
    data_folder: str = "GermanSolarFarm/data/"
    feature_cols: List[str] = field(default_factory=lambda: [
        "sunposition_solarHeight",
        "Albedo",
        "TemperatureAt0",
        "RelativeHumidityAt0",
        "SnowDensityAt0",
        "SurfacePressureAt0",
        "SolarRadiationGlobalAt0",
        "hour_of_day_sin",
    ])
    target_col: str = "power_normed"
    n_top_features: int = 8
    test_size: float = 0.2
    shuffle: bool = False
    random_state: int = 42
    shap_sample_size: int = 100
    csv_delimiter: str = ";"
    results_dir: str = "results"
    n_clusters: int = 3
    power_scale: float = 5.5


@dataclass
class ModelConfig:
    # Match the notebook (XCFL3.ipynb) hyperparameters exactly
    n_estimators: int = 200
    max_depth: int = 6
    learning_rate: float = 0.05
    random_state: int = 42
    verbosity: int = 0
