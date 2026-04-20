# XCFL – Explainable Clustered Federated Learning for Solar Energy Forecasting

A Python implementation of the **XCFL** framework introduced in:

> *Explainable Clustered Federated Learning for Solar Energy Forecasting*  
> Energies, 2025, 18(9), 2380 — https://www.mdpi.com/1996-1073/18/9/2380

XCFL extends Clustered Federated Learning (CFL) with **SHAP-based explainability**: each client's contribution to the global model is weighted by both its dataset size and the aggregate SHAP feature-importance contribution score to its clustered model. This makes the aggregation process transparent and data-driven.

---

## Project Structure

```
xcfl_solar/
├── main.py                   # Entry point – runs XCFL, FedAvg, and Centralized
│
├── xcfl/                     # Core XCFL package
│   ├── config.py             # XCFLConfig and ModelConfig dataclasses
│   ├── data_loader.py        # CSV loading utilities
│   ├── client.py             # FederatedClient class
│   ├── clustering.py         # ClientClusterer (Mean Shift)
│   ├── server.py             # XCFLServer (two-level weighted aggregation)
│   └── evaluation.py        # Metrics: RMSE, MAE, R²
│
├── baselines/                # Comparison methods
│   ├── fedavg.py             # FedAvgServer
│   └── centralized.py       # CentralizedModel
│
├── results/                  # Generated CSVs written here at runtime
├── requirements.txt
└── .gitignore
```

---

## Method Overview

### XCFL Pipeline

```
1. Load client CSVs          → one CSV per photovoltaic (PV) farm
2. Build representations     → per-client mean feature vectors
3. Cluster clients           → Mean Shift on standardized representations
4. Train local models        → XGBoost per client
5. Compute SHAP scores       → scalar importance score per client
6. Compute XCFL weights      → w_i ∝ size_i × shap_i
7. Cluster-level aggregation → weighted average within each cluster
8. Global aggregation        → weighted average across clusters
9. Evaluate                  → RMSE, MAE, R²
```

### Client Weighting Formula

```
w_i = (size_i / Σ size) × (shap_i / Σ shap)
w_i = w_i / Σ w          (renormalize)
```

---

## Installation

```bash
git clone https://github.com/<your-username>/xcfl-solar.git
cd xcfl-solar
pip install -r requirements.txt
```

**Python 3.10+** is required (uses `X | Y` union type hints).

---

## Dataset

This implementation uses the **German Solar Farm** dataset (21 PV facilities).

Place the CSV files at:

```
GermanSolarFarm/
└── data/
    ├── pv_01.csv
    ├── pv_02.csv
    └── ...  (pv_01 – pv_21)
```

Each CSV uses `;` as the delimiter and must contain the following columns:

| Column | Description |
|--------|-------------|
| `sunposition_solarHeight` | Solar elevation angle |
| `Albedo` | Surface albedo |
| `TemperatureAt0` | Air temperature at ground level |
| `RelativeHumidityAt0` | Relative humidity |
| `SnowDensityAt0` | Snow density |
| `SurfacePressureAt0` | Surface pressure |
| `SolarRadiationGlobalAt0` | Global solar radiation |
| `power_normed` | Normalized PV output **(target)** |

---

## Usage

```bash
python main.py
```

Results are written to `results/`:

| File | Contents |
|------|----------|
| `comparison.csv` | RMSE / MAE / R² for all three methods |
| `xcfl_metrics.csv` | XCFL metrics |
| `fedavg_metrics.csv` | FedAvg metrics |
| `centralized_metrics.csv` | Centralized metrics |
| `xcfl_client_weights.csv` | Per-client XCFL weights and cluster assignments |

---

## Configuration

Edit the dataclasses in `xcfl/config.py` to change default settings:

```python
@dataclass
class XCFLConfig:
    data_folder: str = "GermanSolarFarm/data/"
    test_size: float = 0.2
    shap_sample_size: int = 100   # samples used for SHAP computation per client
    ...

@dataclass
class ModelConfig:
    n_estimators: int = 200
    max_depth: int = 6
    learning_rate: float = 0.05
    ...
```

---

## Using the Classes Directly

```python
from xcfl import XCFLConfig, ModelConfig, FederatedClient, ClientClusterer, XCFLServer
from xcfl.data_loader import load_client_files
import os, pandas as pd

cfg = XCFLConfig()
mcfg = ModelConfig()

# Build clients
csv_files = load_client_files(cfg.data_folder)
clients = []
for fname in csv_files:
    c = FederatedClient(fname.replace(".csv", ""), os.path.join(cfg.data_folder, fname), cfg, mcfg)
    c.setup()          # load + train + SHAP in one call
    clients.append(c)

# Cluster
clusterer = ClientClusterer()
labels = clusterer.fit(clients)

# Aggregate
X_test = pd.concat([c.X_test for c in clients], ignore_index=True)
y_test = pd.concat([c.y_test for c in clients], ignore_index=True)

server = XCFLServer(clients, labels)
predictions = server.run(X_test)
```

---

## Citation

```bibtex
@article{xcfl2025,
  title   = {Explainable Clustered Federated Learning for Solar Energy Forecasting},
  journal = {Energies},
  volume  = {18},
  number  = {9},
  pages   = {2380},
  year    = {2025},
  doi     = {10.3390/en18092380},
  url     = {https://www.mdpi.com/1996-1073/18/9/2380}
}
```
