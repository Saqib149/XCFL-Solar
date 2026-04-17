import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Optional

from .client import FederatedClient


class XCFLServer:
    """
    XCFL aggregation server.

    Implements the two-level weighted aggregation from the paper:

      1. Compute per-client weights:   w_i = (size_i / Σsize) * (shap_i / Σshap)
         then renormalize so Σw_i = 1.

      2. Cluster-level aggregation:    cluster_pred[k] = Σ_{i∈k} w_i * model_i(X)

      3. Global aggregation:           final_pred = Σ_k (W_k / ΣW) * cluster_pred[k]
         where W_k = Σ_{i∈k} w_i.

    Because the w_i already sum to 1, steps 2 and 3 together reduce to a single
    weighted average, but the two-level structure mirrors the cluster-aware
    update process described in the paper.
    """

    def __init__(
        self,
        clients: List[FederatedClient],
        cluster_labels: np.ndarray,
    ) -> None:
        if len(clients) != len(cluster_labels):
            raise ValueError("clients and cluster_labels must have the same length.")
        self.clients = clients
        self.cluster_labels = cluster_labels

        self._weights: Optional[np.ndarray] = None
        self._cluster_weights: Optional[Dict[int, List[float]]] = None
        self._cluster_predictions: Optional[Dict[int, np.ndarray]] = None

    # ------------------------------------------------------------------
    # Step 1 – Weight computation
    # ------------------------------------------------------------------

    def compute_xcfl_weights(self) -> np.ndarray:
        """
        Compute and store XCFL client weights.

        w_i ∝ (normalized dataset size) × (normalized SHAP score)
        """
        sizes = np.array([c.dataset_size for c in self.clients], dtype=float)
        shap_scores = np.array([c.shap_score for c in self.clients], dtype=float)

        sizes /= sizes.sum()
        shap_scores /= shap_scores.sum()

        raw = sizes * shap_scores
        self._weights = raw / raw.sum()
        return self._weights

    # ------------------------------------------------------------------
    # Step 2 – Cluster-level aggregation
    # ------------------------------------------------------------------

    def cluster_aggregate(self, X_test: pd.DataFrame) -> Dict[int, np.ndarray]:
        """
        Produce one prediction array per cluster via weighted model averaging.

        Must call compute_xcfl_weights() first.
        """
        self._assert_weights()

        cluster_client_map: Dict[int, List[FederatedClient]] = defaultdict(list)
        cluster_weight_map: Dict[int, List[float]] = defaultdict(list)

        for i, label in enumerate(self.cluster_labels):
            cluster_client_map[int(label)].append(self.clients[i])
            cluster_weight_map[int(label)].append(float(self._weights[i]))

        self._cluster_weights = dict(cluster_weight_map)

        self._cluster_predictions = {}
        for label, members in cluster_client_map.items():
            preds = np.zeros(len(X_test))
            for client, w in zip(members, cluster_weight_map[label]):
                preds += w * client.predict(X_test)
            self._cluster_predictions[label] = preds

        return dict(self._cluster_predictions)

    # ------------------------------------------------------------------
    # Step 3 – Global aggregation
    # ------------------------------------------------------------------

    def global_aggregate(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Combine cluster predictions into a single global prediction.

        Must call cluster_aggregate() first.
        """
        self._assert_cluster_preds()

        cluster_total_w = {
            label: sum(ws) for label, ws in self._cluster_weights.items()
        }
        total_w = sum(cluster_total_w.values())
        cluster_level_w = {label: w / total_w for label, w in cluster_total_w.items()}

        final = np.zeros(len(X_test))
        for label, pred in self._cluster_predictions.items():
            final += cluster_level_w[label] * pred

        return final

    # ------------------------------------------------------------------
    # Convenience: run all three steps in sequence
    # ------------------------------------------------------------------

    def run(self, X_test: pd.DataFrame) -> np.ndarray:
        """Compute weights, cluster-aggregate, then globally aggregate."""
        self.compute_xcfl_weights()
        self.cluster_aggregate(X_test)
        return self.global_aggregate(X_test)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def weight_summary(self) -> pd.DataFrame:
        """Return a DataFrame with per-client weights and cluster assignments."""
        self._assert_weights()
        return pd.DataFrame({
            "client_id": [c.client_id for c in self.clients],
            "cluster": self.cluster_labels,
            "dataset_size": [c.dataset_size for c in self.clients],
            "shap_score": [c.shap_score for c in self.clients],
            "xcfl_weight": self._weights,
        })

    def cluster_weight_summary(self) -> pd.DataFrame:
        """Return a DataFrame with total weight per cluster."""
        self._assert_cluster_preds()
        rows = [
            {"cluster": label, "total_weight": sum(ws)}
            for label, ws in self._cluster_weights.items()
        ]
        return pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assert_weights(self) -> None:
        if self._weights is None:
            raise RuntimeError("Call compute_xcfl_weights() first.")

    def _assert_cluster_preds(self) -> None:
        if self._cluster_predictions is None:
            raise RuntimeError("Call cluster_aggregate() first.")
