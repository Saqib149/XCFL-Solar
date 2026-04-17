"""
XCFL – Explainable Clustered Federated Learning for Solar Energy Forecasting.

Paper: https://www.mdpi.com/1996-1073/18/9/2380
"""

from .config import XCFLConfig, ModelConfig
from .client import FederatedClient
from .clustering import ClientClusterer
from .server import XCFLServer
from .evaluation import compute_metrics, print_metrics, save_metrics, compare_methods

__all__ = [
    "XCFLConfig",
    "ModelConfig",
    "FederatedClient",
    "ClientClusterer",
    "XCFLServer",
    "compute_metrics",
    "print_metrics",
    "save_metrics",
    "compare_methods",
]
