"""
backend/graph_utils.py

Graph construction and model inference utilities.

STUB FILE — ML team handoff interface.

All functions here are stubs with documented return shapes.
ML team: replace the stub bodies ONLY. Do not change:
  - Function signatures
  - Return shapes (these are the API contract with xai_bridge.py and main.py)
  - COW_IDS list (must stay in sync with mock_data.py)

Expected ML team deliverables:
  - backend/models/tauron_model.pt  (GraphSAGE+GRU weights via torch.save)
  - Real bodies for build_graph(), run_inference(), get_gnn_explainer_output()
  - Confirmation that FEATURE_NAMES in xai_bridge.py matches training feature order
"""

import numpy as np

# Demo farm roster — must match mock_data.py cow IDs
COW_IDS = [47, 31, 22, 8, 15, 3, 9, 27]

# Path for the trained model weights
MODEL_PATH = "backend/models/tauron_model.pt"


def build_graph(farm_data: dict | None = None):
    """
    Construct a PyTorch Geometric Data object from farm records.

    Args:
        farm_data: dict with keys:
            "pen_assignments": list of (cow_id, pen_id) tuples
            "milk_yields":     list of (cow_id, yield_kg, timestamp) tuples
            "vet_events":      list of (cow_id, event_type, timestamp) tuples
          If None, uses synthetic demo data.

    Returns:
        torch_geometric.data.Data with attributes:
            x          (torch.Tensor): node feature matrix, shape [num_cows, num_features]
                                       feature order must match FEATURE_NAMES in xai_bridge.py
            edge_index (torch.Tensor): COO edge list, shape [2, num_edges], dtype=long
            edge_attr  (torch.Tensor): edge weights, shape [num_edges], dtype=float32
            cow_ids    (list[int]):    maps node row index → cow ID (e.g. [47, 31, 22, ...])

    ML Team: Replace this stub body. Signature is fixed.
    """
    raise NotImplementedError(
        "build_graph() stub — ML team has not yet integrated the real implementation. "
        "Set USE_MOCK = True in mock_data.py to use hardcoded demo data."
    )


def run_inference(graph_data) -> dict:
    """
    Run the GraphSAGE+GRU forward pass and return herd risk scores.

    Args:
        graph_data: torch_geometric.data.Data object from build_graph()

    Returns:
        dict matching the /herd response schema exactly:
        {
            "cows": [
                {
                    "id": int,
                    "risk_score": float,      # [0.0, 1.0]
                    "status": str,            # "alert" | "watch" | "ok"
                    "top_feature": str | None # highest-importance feature, or None if ok
                },
                ...
            ],
            "adjacency": list[list[int]]  # N×N matrix, row/col order = cows list order
        }

    Status thresholds:
        risk_score > 0.70  → "alert"
        risk_score 0.40–0.70 → "watch"
        risk_score < 0.40  → "ok"

    ML Team: Replace this stub body. Return shape is fixed — do not add/remove keys.
    """
    raise NotImplementedError(
        "run_inference() stub — ML team has not yet integrated the real implementation. "
        "Set USE_MOCK = True in mock_data.py to use hardcoded demo data."
    )


def get_gnn_explainer_output(cow_id: int, graph_data) -> dict:
    """
    Run GNNExplainer for a specific cow and return raw explanation masks.

    Only call this for cows with risk_score > 0.70 (alert status) to
    avoid unnecessary computation. The xai_bridge.py handles this gate.

    Args:
        cow_id:     the cow's farm ID (e.g. 47)
        graph_data: torch_geometric.data.Data object from build_graph()

    Returns:
        {
            "cow_id":       int,
            "edge_mask":    list[float],       # importance score per edge, range [0, 1]
                                               # same length and order as edge_index
            "edge_index":   list[list[int]],   # [[from_node_idx, to_node_idx], ...]
                                               # node indices (not cow IDs) into cow_ids list
            "feature_mask": list[float],       # importance score per feature, range [0, 1]
                                               # order must match FEATURE_NAMES in xai_bridge.py
            "feature_names": list[str]         # feature names in same order as feature_mask
        }

    ML Team: Replace this stub body. Return shape is fixed — do not add/remove keys.
             edge_index must use node indices (row positions in graph_data.x), NOT cow IDs.
    """
    raise NotImplementedError(
        "get_gnn_explainer_output() stub — ML team has not yet integrated the real implementation. "
        "Set USE_MOCK = True in mock_data.py to use hardcoded demo data."
    )
