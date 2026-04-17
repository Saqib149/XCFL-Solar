"""
main.py – Run the XCFL experiment and compare against FedAvg and Centralized baselines.

Usage
-----
    python main.py

The script expects the GermanSolarFarm dataset to be present at the path
configured in XCFLConfig.data_folder (default: GermanSolarFarm/data/).
"""

import os
import pandas as pd

from xcfl import (
    XCFLConfig,
    ModelConfig,
    FederatedClient,
    ClientClusterer,
    XCFLServer,
    compute_metrics,
    print_metrics,
    save_metrics,
    compare_methods,
)
from xcfl.data_loader import load_client_files
from baselines import FedAvgServer, CentralizedModel


def build_clients(
    csv_files: list,
    xcfl_config: XCFLConfig,
    model_config: ModelConfig,
) -> list[FederatedClient]:
    clients = []
    for i, fname in enumerate(csv_files):
        client = FederatedClient(
            client_id=fname.replace(".csv", ""),
            file_path=os.path.join(xcfl_config.data_folder, fname),
            xcfl_config=xcfl_config,
            model_config=model_config,
        )
        client.load_and_split()
        clients.append(client)
    return clients


def run_xcfl(clients: list[FederatedClient], X_test_combined: pd.DataFrame, y_test_combined: pd.Series) -> dict:
    print("\n[XCFL] Training local models + computing SHAP scores …")
    for i, client in enumerate(clients):
        print(f"  client {i + 1}/{len(clients)}: {client.client_id}")
        client.train()
        client.compute_shap_score()

    print("\n[XCFL] Clustering clients …")
    clusterer = ClientClusterer()
    cluster_labels = clusterer.fit(clients)
    print(f"  Cluster assignments: {cluster_labels}")
    print(f"  Number of clusters : {clusterer.n_clusters}")

    print("\n[XCFL] Aggregating predictions …")
    server = XCFLServer(clients, cluster_labels)
    final_preds = server.run(X_test_combined)

    metrics = compute_metrics(y_test_combined, final_preds)
    print_metrics(metrics, label="XCFL Results")

    weight_df = server.weight_summary()
    os.makedirs("results", exist_ok=True)
    weight_df.to_csv("results/xcfl_client_weights.csv", index=False)
    print("  Client weights saved → results/xcfl_client_weights.csv")

    return metrics


def run_fedavg(clients: list[FederatedClient], X_test_combined: pd.DataFrame, y_test_combined: pd.Series) -> dict:
    print("\n[FedAvg] Running FedAvg baseline …")
    server = FedAvgServer(clients)
    final_preds = server.run(X_test_combined)

    metrics = compute_metrics(y_test_combined, final_preds)
    print_metrics(metrics, label="FedAvg Results")
    return metrics


def run_centralized(xcfl_config: XCFLConfig, model_config: ModelConfig) -> dict:
    print("\n[Centralized] Training centralized model …")
    model = CentralizedModel(xcfl_config, model_config)
    model.load_and_train()
    preds = model.predict()

    metrics = compute_metrics(model.y_test, preds)
    print_metrics(metrics, label="Centralized Results")
    return metrics


def combine_test_sets(clients: list[FederatedClient]):
    X_all = pd.concat([c.X_test for c in clients], ignore_index=True)
    y_all = pd.concat([c.y_test for c in clients], ignore_index=True)
    return X_all, y_all


def main() -> None:
    xcfl_config = XCFLConfig()
    model_config = ModelConfig()

    # ------------------------------------------------------------------ #
    # 1. Load client files
    # ------------------------------------------------------------------ #
    print("=" * 60)
    print("  XCFL – Explainable Clustered Federated Learning")
    print("=" * 60)

    csv_files = load_client_files(xcfl_config.data_folder)
    print(f"\nFound {len(csv_files)} client files in '{xcfl_config.data_folder}'")

    # ------------------------------------------------------------------ #
    # 2. Build clients (load + split only; training happens per method)
    # ------------------------------------------------------------------ #
    print("\n[Setup] Loading client datasets …")
    clients = build_clients(csv_files, xcfl_config, model_config)

    X_test_combined, y_test_combined = combine_test_sets(clients)

    # ------------------------------------------------------------------ #
    # 3. Run all three methods
    # ------------------------------------------------------------------ #
    xcfl_metrics = run_xcfl(clients, X_test_combined, y_test_combined)
    fedavg_metrics = run_fedavg(clients, X_test_combined, y_test_combined)
    centralized_metrics = run_centralized(xcfl_config, model_config)

    # ------------------------------------------------------------------ #
    # 4. Compare and save
    # ------------------------------------------------------------------ #
    comparison = compare_methods({
        "XCFL": xcfl_metrics,
        "FedAvg": fedavg_metrics,
        "Centralized": centralized_metrics,
    })

    print("\n\n===== Method Comparison =====")
    print(comparison.to_string())

    os.makedirs("results", exist_ok=True)
    comparison.to_csv("results/comparison.csv")
    save_metrics(xcfl_metrics, "results/xcfl_metrics.csv")
    save_metrics(fedavg_metrics, "results/fedavg_metrics.csv")
    save_metrics(centralized_metrics, "results/centralized_metrics.csv")

    print("\nAll results saved to results/")


if __name__ == "__main__":
    main()
