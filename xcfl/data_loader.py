import os
import pandas as pd
from typing import List


def load_client_files(data_folder: str) -> List[str]:
    """Return sorted list of CSV file names found in data_folder."""
    return sorted(f for f in os.listdir(data_folder) if f.endswith(".csv"))


def load_client_dataframe(
    file_path: str,
    feature_cols: List[str],
    target_col: str,
    delimiter: str = ";",
) -> pd.DataFrame:
    """Load a single client CSV and return only the required columns."""
    df = pd.read_csv(file_path, delimiter=delimiter)
    return df[feature_cols + [target_col]]


def load_combined_dataframe(
    csv_files: List[str],
    data_folder: str,
    feature_cols: List[str],
    target_col: str,
    delimiter: str = ";",
) -> pd.DataFrame:
    """Concatenate all client CSVs into one DataFrame (for centralized baseline)."""
    frames = [
        load_client_dataframe(
            os.path.join(data_folder, f), feature_cols, target_col, delimiter
        )
        for f in csv_files
    ]
    return pd.concat(frames, ignore_index=True)
