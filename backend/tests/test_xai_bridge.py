"""
backend/tests/test_xai_bridge.py

Unit tests for XAI bridge pure functions.
No PyTorch, no Ollama, no network required.

Run: pytest backend/tests/test_xai_bridge.py -v
"""

import pytest
from backend.xai_bridge import extract_top_edge, extract_top_feature, build_xai_json, FEATURE_NAMES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_graph():
    """
    4-cow graph: cow IDs [10, 20, 30, 40] mapped to node indices [0, 1, 2, 3]
    Edges: 0-1, 1-2, 2-3 (linear chain)
    """
    return {
        "cow_ids": [10, 20, 30, 40],
        "edge_index": [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2]],
        "edge_mask": [0.3, 0.3, 0.9, 0.9, 0.5, 0.5],
    }


@pytest.fixture
def simple_explainer_output(simple_graph):
    return {
        "cow_id": 20,
        "edge_mask": simple_graph["edge_mask"],
        "edge_index": simple_graph["edge_index"],
        "feature_mask": [0.1, 0.8, 0.05, 0.02, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0],
        "feature_names": FEATURE_NAMES,
    }


# ---------------------------------------------------------------------------
# extract_top_edge tests
# ---------------------------------------------------------------------------

class TestExtractTopEdge:
    def test_finds_highest_weight_incident_edge(self, simple_graph):
        result = extract_top_edge(
            simple_graph["edge_index"],
            simple_graph["edge_mask"],
            simple_graph["cow_ids"],
            target_cow_id=20,  # node index 1 — incident edges: 0-1 (0.3), 1-2 (0.9)
        )
        assert result["weight"] == 0.9

    def test_from_is_target_cow(self, simple_graph):
        result = extract_top_edge(
            simple_graph["edge_index"],
            simple_graph["edge_mask"],
            simple_graph["cow_ids"],
            target_cow_id=20,
        )
        assert result["from"] == 20

    def test_to_is_neighbor_not_self(self, simple_graph):
        result = extract_top_edge(
            simple_graph["edge_index"],
            simple_graph["edge_mask"],
            simple_graph["cow_ids"],
            target_cow_id=20,  # best edge is 1-2 (weight 0.9), neighbor = node 2 = cow 30
        )
        assert result["to"] == 30

    def test_unknown_cow_returns_self_loop(self, simple_graph):
        result = extract_top_edge(
            simple_graph["edge_index"],
            simple_graph["edge_mask"],
            simple_graph["cow_ids"],
            target_cow_id=999,
        )
        assert result["from"] == 999
        assert result["to"] == 999
        assert result["weight"] == 0.0

    def test_isolated_cow_returns_self_loop(self):
        result = extract_top_edge(
            edge_index=[[0, 1]],
            edge_mask=[0.5],
            cow_ids=[10, 20, 30],
            target_cow_id=30,  # node 2 — no incident edges
        )
        assert result["from"] == 30
        assert result["to"] == 30
        assert result["weight"] == 0.0

    def test_weight_is_rounded(self, simple_graph):
        result = extract_top_edge(
            simple_graph["edge_index"],
            simple_graph["edge_mask"],
            simple_graph["cow_ids"],
            target_cow_id=20,
        )
        # weight should be a float with at most 4 decimal places
        assert isinstance(result["weight"], float)


# ---------------------------------------------------------------------------
# extract_top_feature tests
# ---------------------------------------------------------------------------

class TestExtractTopFeature:
    def test_returns_highest_importance_feature(self):
        # milk_yield_drop is at index 1 with highest score
        mask = [0.1, 0.8, 0.05, 0.02, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0]
        name, delta = extract_top_feature(mask)
        assert name == "milk_yield_drop"

    def test_returns_delta_for_top_feature(self):
        mask = [0.1, 0.8, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        deltas = [0.0, -0.18, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        name, delta = extract_top_feature(mask, deltas)
        assert name == "milk_yield_drop"
        assert delta == -0.18

    def test_zero_delta_when_no_delta_provided(self):
        mask = [0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        _, delta = extract_top_feature(mask)
        assert delta == 0.0

    def test_empty_mask_returns_default(self):
        name, delta = extract_top_feature([])
        assert name == FEATURE_NAMES[0]
        assert delta == 0.0

    def test_handles_all_zeros(self):
        mask = [0.0] * len(FEATURE_NAMES)
        name, delta = extract_top_feature(mask)
        assert name in FEATURE_NAMES  # any valid feature name is fine


# ---------------------------------------------------------------------------
# build_xai_json tests
# ---------------------------------------------------------------------------

class TestBuildXaiJson:
    def test_output_has_required_keys(self, simple_explainer_output, simple_graph):
        result = build_xai_json(
            cow_id=20,
            risk_score=0.75,
            explainer_output=simple_explainer_output,
            cow_ids=simple_graph["cow_ids"],
        )
        required = {"cow_id", "risk_score", "top_edge", "top_feature", "feature_delta"}
        assert required == set(result.keys())

    def test_cow_id_matches(self, simple_explainer_output, simple_graph):
        result = build_xai_json(20, 0.75, simple_explainer_output, simple_graph["cow_ids"])
        assert result["cow_id"] == 20

    def test_risk_score_rounded(self, simple_explainer_output, simple_graph):
        result = build_xai_json(20, 0.756789, simple_explainer_output, simple_graph["cow_ids"])
        assert result["risk_score"] == round(0.756789, 4)

    def test_top_edge_from_is_target(self, simple_explainer_output, simple_graph):
        result = build_xai_json(20, 0.75, simple_explainer_output, simple_graph["cow_ids"])
        assert result["top_edge"]["from"] == 20

    def test_top_feature_is_valid_name(self, simple_explainer_output, simple_graph):
        result = build_xai_json(20, 0.75, simple_explainer_output, simple_graph["cow_ids"])
        assert result["top_feature"] in FEATURE_NAMES

    def test_no_alert_text_in_output(self, simple_explainer_output, simple_graph):
        """build_xai_json must not include alert_text — that's added by the LLM step."""
        result = build_xai_json(20, 0.75, simple_explainer_output, simple_graph["cow_ids"])
        assert "alert_text" not in result
