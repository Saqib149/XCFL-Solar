import numpy as np
import pandas as pd
import xgboost as xgb
from collections import defaultdict
from typing import Dict, List, Optional

from .client import FederatedClient
from .config import ModelConfig


class XCFLServer:
    """
    XCFL aggregation server.

    Algorithm (per the paper):
      1. Compute per-client XCFL weights:  w_i = (size_i * shap_i) / sum(size_j * shap_j)
      2. Cluster clients (KMeans on mean feature vectors).
      3. For each cluster k, train one dedicated XGBoost model on the combined
         training data of all cluster members (equal sample weights — the SHAP
         scores already determined cluster membership via the weight ranking).
      4. At inference time, each client's test data is predicted by its own
         cluster's model only — no cross-cluster contamination.
    """

    def __init__(
        self,
        clients: List[FederatedClient],
        cluster_labels: np.ndarray,
        model_config: ModelConfig,
    ) -> None:
        if len(clients) != len(cluster_labels):
            raise ValueError("clients and cluster_labels must have the same length.")
        self.clients = clients
        self.cluster_labels = cluster_labels
        self.model_config = model_config

        self._weights: Optional[np.ndarray] = None
        self._cluster_models: Optional[Dict[int, xgb.XGBRegressor]] = None

    # ------------------------------------------------------------------
    # Step 1 – Weight computation
    # ------------------------------------------------------------------

    def compute_xcfl_weights(self) -> np.ndarray:
        """w_i ∝ dataset_size_i × shap_score_i, renormalised to sum to 1."""
        sizes = np.array([c.dataset_size for c in self.clients], dtype=float)
        shap_scores = np.array([c.shap_score for c in self.clients], dtype=float)
        raw = sizes * shap_scores
        self._weights = raw / raw.sum()
        return self._weights

    # ------------------------------------------------------------------
    # Step 2 – Train one XGBoost model per cluster
    # ------------------------------------------------------------------

    def train_cluster_models(self) -> Dict[int, xgb.XGBRegressor]:
        """
        Pool training data from all clients in each cluster and fit one
        XGBoost model per cluster with equal sample weights.
        """
        self._assert_weights()

        cluster_idx_map: Dict[int, List[int]] = defaultdict(list)
        for i, label in enumerate(self.cluster_labels):
            cluster_idx_map[int(label)].append(i)

        self._cluster_models = {}
        for cluster_id, indices in cluster_idx_map.items():
            members = [self.clients[i] for i in indices]

            X_combined = pd.concat(
                [m.X_train for m in members], ignore_index=True
            )
            y_combined = pd.concat(
                [m.y_train for m in members], ignore_index=True
            )

            mc = self.model_config
            model = xgb.XGBRegressor(
                n_estimators=mc.n_estimators,
                max_depth=mc.max_depth,
                learning_rate=mc.learning_rate,
                random_state=mc.random_state,
                verbosity=mc.verbosity,
            )
            model.fit(X_combined, y_combined)
            self._cluster_models[cluster_id] = model

        return self._cluster_models

    # ------------------------------------------------------------------
    # Step 3 – Cluster-specialised inference
    # ------------------------------------------------------------------

    def predict_per_client(self) -> np.ndarray:
        """
        Predict each client's test split using its cluster's model.
        Output is concatenated in clients-list order, matching combine_test_sets().
        """
        if self._cluster_models is None:
            raise RuntimeError("Call train_cluster_models() first.")

        all_preds: List[np.ndarray] = []
        for i, client in enumerate(self.clients):
            cluster_id = int(self.cluster_labels[i])
            preds = self._cluster_models[cluster_id].predict(client.X_test)
            all_preds.append(preds)
        return np.concatenate(all_preds)

    # ------------------------------------------------------------------
    # Convenience: run all steps in sequence
    # ------------------------------------------------------------------

    def run(self) -> np.ndarray:
        """Compute weights → train cluster models → predict per client."""
        self.compute_xcfl_weights()
        self.train_cluster_models()
        return self.predict_per_client()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def weight_summary(self) -> pd.DataFrame:
        self._assert_weights()
        return pd.DataFrame({
            "client_id": [c.client_id for c in self.clients],
            "cluster": self.cluster_labels,
            "dataset_size": [c.dataset_size for c in self.clients],
            "shap_score": [c.shap_score for c in self.clients],
            "xcfl_weight": self._weights,
        })

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assert_weights(self) -> None:
        if self._weights is None:
            raise RuntimeError("Call compute_xcfl_weights() first.")
