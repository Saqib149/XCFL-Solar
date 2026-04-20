import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import List, Optional

from .client import FederatedClient


class ClientClusterer:
    """
    Groups federated clients by data similarity.

    Uses KMeans when n_clusters is specified (recommended for reproducibility),
    otherwise falls back to MeanShift auto-detection.
    """

    def __init__(self, n_clusters: Optional[int] = None) -> None:
        self.n_clusters = n_clusters
        self._scaler = StandardScaler()
        self._labels: Optional[np.ndarray] = None

    def fit(self, clients: List[FederatedClient]) -> np.ndarray:
        representations = np.array([c.representation for c in clients])
        normalized = self._scaler.fit_transform(representations)

        if self.n_clusters is not None:
            clusterer = KMeans(
                n_clusters=self.n_clusters,
                random_state=42,
                n_init=10,
            )
        else:
            from sklearn.cluster import MeanShift
            clusterer = MeanShift()

        self._labels = clusterer.fit_predict(normalized)
        return self._labels

    @property
    def labels(self) -> np.ndarray:
        if self._labels is None:
            raise RuntimeError("Call fit() before accessing labels.")
        return self._labels

    @property
    def n_clusters_found(self) -> int:
        return int(np.unique(self._labels).size)
