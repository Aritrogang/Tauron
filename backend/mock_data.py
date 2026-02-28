# backend/mock_data.py
# FROZEN SCHEMA — do not change key names after frontend team starts building D3.js components.
#
# status enum: "alert" (risk > 0.70), "watch" (0.40–0.70), "ok" (< 0.40)
# top_feature: null for healthy cows — frontend must handle null
# adjacency: N×N matrix, row/col order matches the cows list order exactly
#
# USE_MOCK: flip to False when the ML model is integrated (see graph_utils.py)

USE_MOCK = True

MOCK_HERD = {
    "cows": [
        {"id": 47, "risk_score": 0.85, "status": "alert", "top_feature": "milk_yield_drop"},
        {"id": 31, "risk_score": 0.62, "status": "watch", "top_feature": "proximity_to_alert"},
        {"id": 22, "risk_score": 0.55, "status": "watch", "top_feature": "brd_cough_frequency"},
        {"id": 8,  "risk_score": 0.21, "status": "ok",    "top_feature": None},
        {"id": 15, "risk_score": 0.74, "status": "alert", "top_feature": "lameness_score"},
        {"id": 3,  "risk_score": 0.18, "status": "ok",    "top_feature": None},
        {"id": 9,  "risk_score": 0.41, "status": "watch", "top_feature": "rumination_drop"},
        {"id": 27, "risk_score": 0.33, "status": "ok",    "top_feature": None},
    ],
    # Adjacency matrix: index matches cows list above
    # cows: [47, 31, 22, 8, 15, 3, 9, 27]
    "adjacency": [
        # 47  31  22   8  15   3   9  27
        [  0,  1,  1,  0,  0,  0,  0,  0],  # 47
        [  1,  0,  0,  1,  0,  0,  0,  0],  # 31
        [  1,  0,  0,  0,  1,  0,  1,  0],  # 22
        [  0,  1,  0,  0,  0,  1,  0,  0],  # 8
        [  0,  0,  1,  0,  0,  0,  0,  1],  # 15
        [  0,  0,  0,  1,  0,  0,  1,  0],  # 3
        [  0,  0,  1,  0,  0,  1,  0,  1],  # 9
        [  0,  0,  0,  0,  1,  0,  1,  0],  # 27
    ],
}

MOCK_EXPLAIN = {
    47: {
        "cow_id": 47,
        "risk_score": 0.85,
        "top_edge": {"from": 47, "to": 31, "weight": 0.91},
        "top_feature": "milk_yield_drop",
        "feature_delta": -0.18,
        "alert_text": (
            "Isolate #47: milk yield dropped 18% and she shared Bunk C "
            "with #31, who was treated Tuesday. Check temperature now."
        ),
    },
    31: {
        "cow_id": 31,
        "risk_score": 0.62,
        "top_edge": {"from": 31, "to": 47, "weight": 0.91},
        "top_feature": "proximity_to_alert",
        "feature_delta": 0.0,
        "alert_text": (
            "Monitor #31: shared Bunk C with high-risk Cow #47. "
            "No symptoms yet — recheck at next milking."
        ),
    },
    22: {
        "cow_id": 22,
        "risk_score": 0.55,
        "top_edge": {"from": 22, "to": 47, "weight": 0.76},
        "top_feature": "brd_cough_frequency",
        "feature_delta": 0.12,
        "alert_text": (
            "Watch #22: cough frequency up 12%, contact with #47. "
            "Separate from herd and monitor breathing."
        ),
    },
    8: {
        "cow_id": 8,
        "risk_score": 0.21,
        "top_edge": {"from": 8, "to": 31, "weight": 0.35},
        "top_feature": "proximity_to_alert",
        "feature_delta": 0.0,
        "alert_text": "No action needed for #8 — low risk, routine monitoring only.",
    },
    15: {
        "cow_id": 15,
        "risk_score": 0.74,
        "top_edge": {"from": 15, "to": 22, "weight": 0.82},
        "top_feature": "lameness_score",
        "feature_delta": 0.25,
        "alert_text": (
            "Isolate #15: lameness score elevated 25%, shared feed station "
            "with #22. Call vet for hoof inspection."
        ),
    },
    3: {
        "cow_id": 3,
        "risk_score": 0.18,
        "top_edge": {"from": 3, "to": 8, "weight": 0.22},
        "top_feature": "proximity_to_alert",
        "feature_delta": 0.0,
        "alert_text": "No action needed for #3 — healthy, continue normal management.",
    },
    9: {
        "cow_id": 9,
        "risk_score": 0.41,
        "top_edge": {"from": 9, "to": 22, "weight": 0.64},
        "top_feature": "rumination_drop",
        "feature_delta": -0.09,
        "alert_text": (
            "Watch #9: rumination time dropped 9%, near #22. "
            "Check feed intake and body condition."
        ),
    },
    27: {
        "cow_id": 27,
        "risk_score": 0.33,
        "top_edge": {"from": 27, "to": 15, "weight": 0.48},
        "top_feature": "proximity_to_alert",
        "feature_delta": 0.0,
        "alert_text": "Monitor #27: moderate contact with #15. No immediate action required.",
    },
}
