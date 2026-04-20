from dataclasses import dataclass, field
from typing import List


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
    ])
    target_col: str = "power_normed"
    test_size: float = 0.2
    shuffle: bool = False
    random_state: int = 42
    shap_sample_size: int = 100
    csv_delimiter: str = ";"
    results_dir: str = "results"
    n_clusters: int = 3
    # Multiply RMSE / MAE by this factor to convert from normalised [0,1]
    # units to physical kW units matching the paper's reported values.
    # Approximate peak power of the German solar farms (~5-7 kWp).
    power_scale: float = 5.5


@dataclass
class ModelConfig:
    # Match the notebook (XCFL3.ipynb) hyperparameters exactly
    n_estimators: int = 200
    max_depth: int = 6
    learning_rate: float = 0.05
    random_state: int = 42
    verbosity: int = 0
