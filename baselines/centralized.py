import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from typing import Optional

from xcfl.config import ModelConfig, XCFLConfig
from xcfl.data_loader import load_combined_dataframe, load_client_files


class CentralizedModel:
    """Centralized training baseline — single XGBoost model trained on all client data combined."""

    def __init__(self, xcfl_config: XCFLConfig, model_config: ModelConfig) -> None:
        self.xcfl_config = xcfl_config
        self.model_config = model_config

        self._model: Optional[xgb.XGBRegressor] = None
        self._X_test: Optional[pd.DataFrame] = None
        self._y_test: Optional[pd.Series] = None

    def load_and_train(self) -> None:
        cfg = self.xcfl_config
        csv_files = load_client_files(cfg.data_folder)
        df_all = load_combined_dataframe(
            csv_files, cfg.data_folder, cfg.feature_cols, cfg.target_col, cfg.csv_delimiter
        )
        X = df_all.drop(columns=[cfg.target_col])
        y = df_all[cfg.target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=cfg.test_size, shuffle=cfg.shuffle
        )
        self._X_test = X_test
        self._y_test = y_test

        mc = self.model_config
        self._model = xgb.XGBRegressor(
            n_estimators=mc.n_estimators,
            max_depth=mc.max_depth,
            learning_rate=mc.learning_rate,
            random_state=mc.random_state,
            verbosity=mc.verbosity,
        )
        self._model.fit(X_train, y_train)

    def predict(self, X: Optional[pd.DataFrame] = None) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call load_and_train() first.")
        return self._model.predict(X if X is not None else self._X_test)

    @property
    def X_test(self) -> pd.DataFrame:
        return self._X_test

    @property
    def y_test(self) -> pd.Series:
        return self._y_test
