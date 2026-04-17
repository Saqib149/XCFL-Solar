import numpy as np
from sklearn.cluster import MeanShift
from sklearn.preprocessing import StandardScaler
from typing import List, Optional

from .client import FederatedClient


class ClientClusterer:
    """
    Groups federated clients by data similarity using Mean Shift clustering.

    Client representations (mean feature vectors) are standardized before
    clustering so that features on different scales contribute equally.
    """

    def __init__(self, bandwidth: Optional[float] = None) -> None:
        self.bandwidth = bandwidth
        self._scaler = StandardScaler()
        self._clusterer: Optional[MeanShift] = None
        self._labels: Optional[np.ndarray] = None

    def fit(self, clients: List[FederatedClient]) -> np.ndarray:
        """
        Fit Mean Shift clustering on client representations.

        Parameters
        ----------
        clients : list of FederatedClient
            Each client must have its data loaded (representation available).

        Returns
        -------
        labels : np.ndarray of shape (n_clients,)
            Integer cluster assignment for each client.
        """
        representations = np.array([c.representation for c in clients])
        normalized = self._scaler.fit_transform(representations)

        self._clusterer = MeanShift(bandwidth=self.bandwidth)
        self._labels = self._clusterer.fit_predict(normalized)
        return self._labels

    @property
    def labels(self) -> np.ndarray:
        if self._labels is None:
            raise RuntimeError("Call fit() before accessing labels.")
        return self._labels

    @property
    def n_clusters(self) -> int:
        return int(np.unique(self._labels).size)

    @property
    def cluster_centers(self) -> np.ndarray:
        """Cluster centers in the original (un-scaled) feature space."""
        return self._scaler.inverse_transform(self._clusterer.cluster_centers_)

    def summary(self) -> dict:
        """Return a dict mapping cluster id -> list of client indices."""
        result: dict = {}
        for idx, label in enumerate(self._labels):
            result.setdefault(int(label), []).append(idx)
        return result
