import numpy as np
import pandas as pd
from typing import List

from xcfl.client import FederatedClient


class FedAvgServer:
    """Federated Averaging (FedAvg) baseline — weights clients by training dataset size."""

    def __init__(self, clients: List[FederatedClient]) -> None:
        self.clients = clients
        self._weights: np.ndarray | None = None

    def compute_weights(self) -> np.ndarray:
        """Weight each client by its fraction of the total training data."""
        sizes = np.array([c.dataset_size for c in self.clients], dtype=float)
        self._weights = sizes / sizes.sum()
        return self._weights

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Weighted-average prediction across all local models."""
        if self._weights is None:
            raise RuntimeError("Call compute_weights() before predict().")
        preds = np.zeros(len(X))
        for client, w in zip(self.clients, self._weights):
            preds += w * client.predict(X)
        return preds

    def run(self, X: pd.DataFrame) -> np.ndarray:
        """Compute weights then predict in one call."""
        self.compute_weights()
        return self.predict(X)

    def weight_summary(self) -> pd.DataFrame:
        if self._weights is None:
            raise RuntimeError("Call compute_weights() first.")
        return pd.DataFrame({
            "client_id": [c.client_id for c in self.clients],
            "dataset_size": [c.dataset_size for c in self.clients],
            "fedavg_weight": self._weights,
        })
