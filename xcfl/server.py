import numpy as np
import pandas as pd
import xgboost as xgb
from collections import defaultdict
from typing import Dict, List, Optional

from .client import FederatedClient
from .config import ModelConfig


class XCFLServer:
    """XCFL aggregation server: computes SHAP-weighted aggregation and trains cluster-specific models."""

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
        """
        Compute per-client weights using Eq. 2 (paper):

            w_k = Σ_f [ δ_f^(k) / Σ_{k'∈Cm} δ_f^(k') ] * [ 1 - Σ_{j≠f} a_j ]

        where a_j = Σ_{k∈Cm} δ_j^(k) / Σ_{j'} Σ_{k∈Cm} δ_{j'}^(k)
        is the normalized cluster-level importance of feature j.
        Weights are computed independently per cluster then assembled.
        """
        weights = np.zeros(len(self.clients))

        cluster_idx_map: dict = defaultdict(list)
        for i, label in enumerate(self.cluster_labels):
            cluster_idx_map[int(label)].append(i)

        for indices in cluster_idx_map.values():
            members = [self.clients[i] for i in indices]

            # delta: shape (K_m, F) — δ_f^(k) per client per feature
            delta = np.array([c.shap_per_feature for c in members], dtype=float)

            # a_j: cluster-level normalized feature importance, shape (F,)
            cluster_delta = delta.sum(axis=0)                          # Σ_{k} δ_j^(k)
            total = cluster_delta.sum()
            a = cluster_delta / total if total > 0 else np.ones(delta.shape[1]) / delta.shape[1]

            # redundancy term per feature: 1 - Σ_{j≠f} a_j = a_f
            redundancy = a                                              # shape (F,)

            # normalized SHAP per client per feature: δ_f^(k) / Σ_{k'} δ_f^(k')
            denom = cluster_delta.copy()
            denom[denom == 0] = 1.0                                    # avoid div-by-zero
            norm_shap = delta / denom                                  # shape (K_m, F)

            # w_k = Σ_f  norm_shap[k,f] * redundancy[f]
            client_weights = (norm_shap * redundancy).sum(axis=1)      # shape (K_m,)

            for local_i, global_i in enumerate(indices):
                weights[global_i] = client_weights[local_i]

        self._weights = weights
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
