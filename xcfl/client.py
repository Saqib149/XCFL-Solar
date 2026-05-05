import os
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split

from .config import XCFLConfig, ModelConfig
from .data_loader import load_client_dataframe


class FederatedClient:
    """
    Represents a single federated learning participant.

    Each client:
      - Loads its own local dataset from a CSV file.
      - Trains a local XGBoost model.
      - Computes a SHAP-based feature importance score used for XCFL weighting.
      - Exposes a representation vector (feature means) for clustering.
    """

    def __init__(
        self,
        client_id: str,
        file_path: str,
        xcfl_config: XCFLConfig,
        model_config: ModelConfig,
    ) -> None:
        self.client_id = client_id
        self.file_path = file_path
        self.xcfl_config = xcfl_config
        self.model_config = model_config

        self._model: xgb.XGBRegressor | None = None
        self._X_train: pd.DataFrame | None = None
        self._X_test: pd.DataFrame | None = None
        self._y_train: pd.Series | None = None
        self._y_test: pd.Series | None = None
        self._shap_score: float | None = None
        self._shap_per_feature: np.ndarray | None = None
        self._representation: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Load data, compute clustering representation, train model, compute SHAP."""
        self._load_and_split()
        self.train()
        self.compute_shap_score()

    def load_and_split(self) -> None:
        self._load_and_split()

    def train(self) -> None:
        """Train a local XGBoost model on this client's training split."""
        if self._X_train is None:
            raise RuntimeError("Call load_and_split() before train().")

        self._model = xgb.XGBRegressor(
            n_estimators=self.model_config.n_estimators,
            max_depth=self.model_config.max_depth,
            learning_rate=self.model_config.learning_rate,
            random_state=self.model_config.random_state,
            verbosity=self.model_config.verbosity,
        )
        self._model.fit(self._X_train, self._y_train)

    def compute_shap_score(self) -> float:
        """
        Compute a scalar SHAP importance score for this client.

        The score is the sum of mean absolute SHAP values across all features,
        sampled from the training set to bound computation time.
        """
        if self._model is None:
            raise RuntimeError("Call train() before compute_shap_score().")

        explainer = shap.TreeExplainer(self._model)
        sample_size = min(self.xcfl_config.shap_sample_size, len(self._X_train))
        X_sample = self._X_train.sample(sample_size, random_state=self.xcfl_config.random_state)
        shap_values = explainer.shap_values(X_sample)

        per_feature = np.abs(shap_values).mean(axis=0)   # δ_f^(k) for each feature f
        self._shap_per_feature = per_feature
        self._shap_score = float(per_feature.sum())
        return self._shap_score

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model has not been trained yet.")
        return self._model.predict(X)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model(self) -> xgb.XGBRegressor:
        return self._model

    @property
    def X_train(self) -> pd.DataFrame:
        return self._X_train

    @property
    def X_test(self) -> pd.DataFrame:
        return self._X_test

    @property
    def y_train(self) -> pd.Series:
        return self._y_train

    @property
    def y_test(self) -> pd.Series:
        return self._y_test

    @property
    def dataset_size(self) -> int:
        """Number of training samples."""
        return len(self._X_train)

    @property
    def shap_score(self) -> float:
        if self._shap_score is None:
            raise RuntimeError("SHAP score not computed yet.")
        return self._shap_score

    @property
    def shap_per_feature(self) -> np.ndarray:
        """Mean absolute SHAP value per feature — δ_f^(k) vector, shape (F,)."""
        if self._shap_per_feature is None:
            raise RuntimeError("SHAP score not computed yet.")
        return self._shap_per_feature

    @property
    def representation(self) -> np.ndarray:
        """Mean feature vector used for client clustering."""
        if self._representation is None:
            raise RuntimeError("Data not loaded yet.")
        return self._representation

    def __repr__(self) -> str:
        trained = self._model is not None
        return (
            f"FederatedClient(id={self.client_id!r}, "
            f"n_train={self.dataset_size if trained else '?'}, trained={trained})"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_and_split(self) -> None:
        cfg = self.xcfl_config
        df = load_client_dataframe(
            self.file_path, cfg.feature_cols, cfg.target_col, cfg.csv_delimiter
        )

        X = df.drop(columns=[cfg.target_col])
        y = df[cfg.target_col]

        self._representation = X.mean().values

        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(
            X, y, test_size=cfg.test_size, shuffle=cfg.shuffle
        )
