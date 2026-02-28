"""
backend/xai_bridge.py

XAI Bridge: transforms raw GNNExplainer output into the /explain/{cow_id} response schema,
then calls the local Mistral-7B LLM to generate a plain-English farmer alert.

Data flow:
    GNNExplainer tensors (from graph_utils.get_gnn_explainer_output)
        → extract_top_edge()        find highest-weight edge incident on target cow
        → extract_top_feature()     find highest-importance node feature
        → build_xai_json()          assemble structured intermediate dict (no alert_text)
        → llm_engine.generate_alert()  Ollama call → alert_text string
        → final response dict       matches /explain/{cow_id} schema exactly

IMPORTANT: FEATURE_NAMES order here must exactly match the feature column order
in the training data used by the ML team. Coordinate this before ML handoff.
If the order is wrong, top_feature will point to the wrong feature.

Pure functions (extract_top_edge, extract_top_feature, build_xai_json) have no side
effects and no ML/network dependencies — fully unit-testable in isolation.
"""

from backend.llm_engine import generate_alert
from backend.mock_data import MOCK_EXPLAIN, USE_MOCK

# Feature names in training column order — must match graph_utils.py feature matrix
# Coordinate with ML team before their handoff
FEATURE_NAMES = [
    "milk_yield",
    "milk_yield_drop",        # delta from 7-day rolling average (negative = drop)
    "vet_event_flag",         # 1 if vet event in past 7 days, else 0
    "pen_assignment",         # pen ID (categorical, encoded)
    "feeding_station",        # feeding station ID (categorical, encoded)
    "rumination_minutes",     # daily rumination time in minutes
    "activity_index",         # step-count / activity score (collar sensor)
    "brd_cough_frequency",    # coughs per hour (acoustic sensor, 0 if no sensor)
    "lameness_score",         # gait score 0–3 (visual assessment, 0 if not assessed)
    "proximity_to_alert",     # 1 if adjacent node has risk_score > 0.70, else 0
]


def extract_top_edge(
    edge_index: list,
    edge_mask: list,
    cow_ids: list,
    target_cow_id: int,
) -> dict:
    """
    Find the edge with the highest GNNExplainer mask weight incident on target cow.

    Args:
        edge_index:     list of [from_node_idx, to_node_idx] pairs (node indices, not cow IDs)
        edge_mask:      list of float importance scores, same length as edge_index
        cow_ids:        list mapping node index → cow ID (e.g. [47, 31, 22, ...])
        target_cow_id:  the cow we're explaining

    Returns:
        {"from": cow_id, "to": neighbor_cow_id, "weight": float}
        Returns self-loop with weight 0.0 if no incident edges found.
    """
    if target_cow_id not in cow_ids:
        return {"from": target_cow_id, "to": target_cow_id, "weight": 0.0}

    target_idx = cow_ids.index(target_cow_id)

    # Filter to edges where target cow is either source or destination
    incident = [
        (i, float(mask_val))
        for i, (edge, mask_val) in enumerate(zip(edge_index, edge_mask))
        if edge[0] == target_idx or edge[1] == target_idx
    ]

    if not incident:
        return {"from": target_cow_id, "to": target_cow_id, "weight": 0.0}

    best_i, best_weight = max(incident, key=lambda x: x[1])
    src_idx, dst_idx = edge_index[best_i]

    # Return the neighbor node (not the target itself)
    neighbor_idx = dst_idx if src_idx == target_idx else src_idx
    neighbor_cow_id = cow_ids[neighbor_idx]

    return {
        "from": target_cow_id,
        "to": neighbor_cow_id,
        "weight": round(best_weight, 4),
    }


def extract_top_feature(
    feature_mask: list,
    feature_delta: list | None = None,
) -> tuple[str, float]:
    """
    Find the most important feature from GNNExplainer's node feature mask.

    Args:
        feature_mask:   list of float importance scores, one per feature
                        order must match FEATURE_NAMES
        feature_delta:  optional list of signed deltas (change from baseline)
                        same order as feature_mask; used for alert context

    Returns:
        (feature_name: str, delta: float)
    """
    if not feature_mask or len(feature_mask) > len(FEATURE_NAMES):
        return FEATURE_NAMES[0], 0.0

    top_idx = max(range(len(feature_mask)), key=lambda i: feature_mask[i])
    top_name = FEATURE_NAMES[top_idx]
    delta = float(feature_delta[top_idx]) if feature_delta and len(feature_delta) > top_idx else 0.0

    return top_name, round(delta, 4)


def build_xai_json(
    cow_id: int,
    risk_score: float,
    explainer_output: dict,
    cow_ids: list,
) -> dict:
    """
    Assemble the structured XAI intermediate dict (without alert_text).

    This is a pure function — no async, no side effects, no ML dependencies.
    Unit-testable with synthetic explainer output.

    This dict is passed directly to llm_engine._build_user_prompt().

    Args:
        cow_id:           farm ID of the cow being explained
        risk_score:       model output risk score [0.0, 1.0]
        explainer_output: dict from graph_utils.get_gnn_explainer_output()
                          keys: cow_id, edge_mask, edge_index, feature_mask, feature_names
        cow_ids:          list mapping node index → cow ID (from graph_utils)

    Returns:
        {
            "cow_id":        int,
            "risk_score":    float,
            "top_edge":      {"from": int, "to": int, "weight": float},
            "top_feature":   str,
            "feature_delta": float
        }
    """
    top_edge = extract_top_edge(
        explainer_output["edge_index"],
        explainer_output["edge_mask"],
        cow_ids,
        cow_id,
    )

    top_feature, feature_delta = extract_top_feature(
        explainer_output["feature_mask"],
        explainer_output.get("feature_delta"),
    )

    return {
        "cow_id": cow_id,
        "risk_score": round(float(risk_score), 4),
        "top_edge": top_edge,
        "top_feature": top_feature,
        "feature_delta": feature_delta,
    }


async def explain_cow(cow_id: int) -> dict:
    """
    Full pipeline: build_graph → GNNExplainer → structured JSON → LLM → response.

    This is the real path called by main.py when USE_MOCK = False.

    Args:
        cow_id: farm ID of the cow to explain

    Returns:
        dict matching the /explain/{cow_id} response schema (including alert_text)

    Raises:
        ValueError: if cow_id not found in inference result
    """
    if USE_MOCK:
        return MOCK_EXPLAIN.get(cow_id, _not_found_response(cow_id))

    # Lazy imports — avoid top-level PyTorch import so server starts without ML deps
    from backend.graph_utils import build_graph, run_inference, get_gnn_explainer_output

    graph = build_graph()
    inference_result = run_inference(graph)

    # Find this cow's risk score in the inference result
    cow_data = next((c for c in inference_result["cows"] if c["id"] == cow_id), None)
    if cow_data is None:
        raise ValueError(f"Cow {cow_id} not found in inference result")

    risk_score = cow_data["risk_score"]
    cow_ids = [c["id"] for c in inference_result["cows"]]

    explainer_output = get_gnn_explainer_output(cow_id, graph)
    xai_json = build_xai_json(cow_id, risk_score, explainer_output, cow_ids)

    alert_text = await generate_alert(xai_json)

    return {**xai_json, "alert_text": alert_text}


def _not_found_response(cow_id: int) -> dict:
    """Safe empty response for unknown cow IDs. Used only in mock mode edge cases."""
    return {
        "cow_id": cow_id,
        "risk_score": 0.0,
        "top_edge": {"from": cow_id, "to": cow_id, "weight": 0.0},
        "top_feature": "unknown",
        "feature_delta": 0.0,
        "alert_text": f"No data available for cow #{cow_id}.",
    }
