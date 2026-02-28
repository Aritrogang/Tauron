# backend/mock_data.py
#
# Demo data for frontend development and emergency rollback.
# Generates a realistic 60-cow herd with contact graph at import time.
#
# Core key names are frozen — the frontend D3.js graph depends on:
#   cows[*].id, cows[*].risk_score, cows[*].status, cows[*].top_feature, adjacency
#
# Pen mapping (used by HerdMap.js): id 1-9 → Pen A, 10-19 → B, ..., 50-60 → F
#
import os
import random as _random

# USE_MOCK: flip to False when tauron_model.pt is trained and graph_utils.py is live.
# Emergency rollback: flip back to True — demo reverts in 30 seconds.
# Also respects USE_MOCK env var (e.g. USE_MOCK=1 uvicorn backend.main:app)

USE_MOCK = os.environ.get("USE_MOCK", "").strip().lower() in ("1", "true", "yes") or False


# ── Generate 60-cow mock herd ──────────────────────────────────────────────

def _build_mock():
    rng = _random.Random(42)  # deterministic for reproducibility

    FEATURES = ["milk_yield_kg", "ear_temp_c", "rumination_min", "activity",
                "feeding_visits", "feeding_min", "days_in_milk"]
    DISEASES = ["mastitis", "brd", "lameness"]

    COW_IDS = list(range(1, 61))  # 60 cows
    N = len(COW_IDS)

    # Hand-picked status — spread across pens for realistic clustering
    alert_ids = {7, 15, 23, 38, 47, 52}            # 6 alert  (~10%)
    watch_ids = {4, 9, 12, 22, 31, 33, 42, 51, 58} # 9 watch  (~15%)
    # remaining 45 cows → ok

    cows = []
    for cid in COW_IDS:
        if cid in alert_ids:
            disease = rng.choice(DISEASES)
            risk = round(rng.uniform(0.72, 0.93), 2)
            other = {d: round(rng.uniform(0.08, 0.35), 2) for d in DISEASES}
            other[disease] = risk
            cows.append({
                "id": cid, "risk_score": risk, "status": "alert",
                "top_feature": rng.choice(FEATURES),
                "dominant_disease": disease, "all_risks": other,
            })
        elif cid in watch_ids:
            disease = rng.choice(DISEASES)
            risk = round(rng.uniform(0.41, 0.69), 2)
            other = {d: round(rng.uniform(0.06, 0.30), 2) for d in DISEASES}
            other[disease] = risk
            cows.append({
                "id": cid, "risk_score": risk, "status": "watch",
                "top_feature": rng.choice(FEATURES),
                "dominant_disease": disease, "all_risks": other,
            })
        else:
            risk = round(rng.uniform(0.03, 0.34), 2)
            cows.append({
                "id": cid, "risk_score": risk, "status": "ok",
                "top_feature": None,
                "dominant_disease": None, "all_risks": None,
            })

    # Override key narrative cows with hand-crafted values
    _overrides = {
        47: {"risk_score": 0.85, "status": "alert", "top_feature": "milk_yield_kg",
             "dominant_disease": "mastitis",
             "all_risks": {"mastitis": 0.85, "brd": 0.31, "lameness": 0.12}},
        15: {"risk_score": 0.74, "status": "alert", "top_feature": "activity",
             "dominant_disease": "lameness",
             "all_risks": {"mastitis": 0.18, "brd": 0.29, "lameness": 0.74}},
        31: {"risk_score": 0.62, "status": "watch", "top_feature": "ear_temp_c",
             "dominant_disease": "brd",
             "all_risks": {"mastitis": 0.41, "brd": 0.62, "lameness": 0.08}},
        22: {"risk_score": 0.55, "status": "watch", "top_feature": "rumination_min",
             "dominant_disease": "brd",
             "all_risks": {"mastitis": 0.22, "brd": 0.55, "lameness": 0.19}},
        9:  {"risk_score": 0.41, "status": "watch", "top_feature": "rumination_min",
             "dominant_disease": "mastitis",
             "all_risks": {"mastitis": 0.41, "brd": 0.28, "lameness": 0.14}},
    }
    for cow in cows:
        if cow["id"] in _overrides:
            cow.update(_overrides[cow["id"]])

    # ── Adjacency matrix ────────────────────────────────────────────────
    # Within-pen: ~55% edge probability (shared stalls/waterers)
    # Cross-pen: ~4% edge probability (milking parlour encounters)
    # Alert neighbours get slightly higher cross-pen connectivity
    adj = [[0] * N for _ in range(N)]
    alert_set = {c["id"] for c in cows if c["status"] == "alert"}

    for i in range(N):
        for j in range(i + 1, N):
            id_i, id_j = COW_IDS[i], COW_IDS[j]
            same_pen = (id_i // 10) == (id_j // 10)
            if same_pen:
                p = 0.55
            elif id_i in alert_set or id_j in alert_set:
                p = 0.07  # alert cows have more cross-pen contacts
            else:
                p = 0.04
            if rng.random() < p:
                adj[i][j] = 1
                adj[j][i] = 1

    return {"cows": cows, "adjacency": adj}


MOCK_HERD = _build_mock()


# ── Explain data — hand-crafted for key cows, generated for the rest ───────

_HAND_EXPLAIN = {
    47: {
        "cow_id": 47, "risk_score": 0.85,
        "top_edge": {"from": 47, "to": 31, "weight": 0.91},
        "top_feature": "milk_yield_kg", "feature_delta": -0.18,
        "dominant_disease": "mastitis",
        "all_risks": {"mastitis": 0.85, "brd": 0.31, "lameness": 0.12},
        "alert_text": (
            "Isolate #47: milk yield dropped 18%, mastitis risk at 85% — "
            "shared pen with #31. Check udder and take temperature now."
        ),
    },
    15: {
        "cow_id": 15, "risk_score": 0.74,
        "top_edge": {"from": 15, "to": 22, "weight": 0.82},
        "top_feature": "activity", "feature_delta": -0.25,
        "dominant_disease": "lameness",
        "all_risks": {"mastitis": 0.18, "brd": 0.29, "lameness": 0.74},
        "alert_text": (
            "Isolate #15: activity dropped 25%, lameness risk at 74% — "
            "shared feed station with #22. Call vet for hoof inspection today."
        ),
    },
    31: {
        "cow_id": 31, "risk_score": 0.62,
        "top_edge": {"from": 31, "to": 47, "weight": 0.91},
        "top_feature": "ear_temp_c", "feature_delta": 0.08,
        "dominant_disease": "brd",
        "all_risks": {"mastitis": 0.41, "brd": 0.62, "lameness": 0.08},
        "alert_text": (
            "Monitor #31: ear temperature up, BRD risk at 62% — "
            "shared space with high-risk #47. Recheck breathing at next milking."
        ),
    },
    22: {
        "cow_id": 22, "risk_score": 0.55,
        "top_edge": {"from": 22, "to": 47, "weight": 0.76},
        "top_feature": "rumination_min", "feature_delta": -0.12,
        "dominant_disease": "brd",
        "all_risks": {"mastitis": 0.22, "brd": 0.55, "lameness": 0.19},
        "alert_text": (
            "Watch #22: rumination time down 12%, BRD risk at 55% — "
            "contact with #47. Separate from herd and monitor breathing."
        ),
    },
    9: {
        "cow_id": 9, "risk_score": 0.41,
        "top_edge": {"from": 9, "to": 22, "weight": 0.64},
        "top_feature": "rumination_min", "feature_delta": -0.09,
        "dominant_disease": "mastitis",
        "all_risks": {"mastitis": 0.41, "brd": 0.28, "lameness": 0.14},
        "alert_text": (
            "Watch #9: rumination time down 9%, mastitis risk at 41% — "
            "near #22. Check feed intake and milk yield at next milking."
        ),
    },
}


def _build_explain():
    """Generate MOCK_EXPLAIN for all 60 cows."""
    rng = _random.Random(99)
    explain = dict(_HAND_EXPLAIN)

    FEATURES = ["milk_yield_kg", "ear_temp_c", "rumination_min", "activity",
                "feeding_visits", "feeding_min"]
    FEATURE_NAMES = {
        "milk_yield_kg": "milk yield", "ear_temp_c": "ear temperature",
        "rumination_min": "rumination time", "activity": "activity",
        "feeding_visits": "feeding visits", "feeding_min": "feeding time",
    }
    DISEASES = ["mastitis", "brd", "lameness"]
    ACTIONS = {
        "mastitis": "Check udder and take temperature at next milking.",
        "brd": "Monitor breathing and check nasal discharge.",
        "lameness": "Inspect hooves and observe gait.",
    }

    cows_by_id = {c["id"]: c for c in MOCK_HERD["cows"]}
    cow_ids = [c["id"] for c in MOCK_HERD["cows"]]
    adj = MOCK_HERD["adjacency"]
    idx_map = {cid: i for i, cid in enumerate(cow_ids)}

    for cid, cow in cows_by_id.items():
        if cid in explain:
            continue

        # Find highest-weight neighbour from adjacency
        i = idx_map[cid]
        neighbours = [cow_ids[j] for j in range(len(adj[i])) if adj[i][j]]
        if neighbours:
            neighbour = rng.choice(neighbours)
            weight = round(rng.uniform(0.3, 0.95), 2)
        else:
            neighbour = cid
            weight = 0.0

        feat = cow["top_feature"] or rng.choice(FEATURES)
        delta = round(rng.uniform(-0.25, 0.05), 2) if cow["status"] != "ok" else 0.0
        disease = cow["dominant_disease"]
        risk = cow["risk_score"]
        risks = cow["all_risks"] or {d: round(rng.uniform(0.03, 0.20), 2) for d in DISEASES}

        if cow["status"] == "alert":
            pct = abs(int(delta * 100))
            text = (
                f"Isolate #{cid}: {FEATURE_NAMES.get(feat, feat)} changed {pct}%, "
                f"{disease} risk at {int(risk * 100)}% — contact with #{neighbour}. "
                f"{ACTIONS.get(disease, 'Monitor closely.')}"
            )
        elif cow["status"] == "watch":
            pct = abs(int(delta * 100))
            text = (
                f"Watch #{cid}: {FEATURE_NAMES.get(feat, feat)} shifted {pct}%, "
                f"{disease} risk at {int(risk * 100)}% — near #{neighbour}. "
                f"{ACTIONS.get(disease, 'Continue monitoring.')}"
            )
        else:
            text = f"No action needed for #{cid} — low risk across all diseases, routine monitoring only."

        explain[cid] = {
            "cow_id": cid, "risk_score": risk,
            "top_edge": {"from": cid, "to": neighbour, "weight": weight},
            "top_feature": feat, "feature_delta": delta,
            "dominant_disease": disease, "all_risks": risks,
            "alert_text": text,
        }

    return explain


MOCK_EXPLAIN = _build_explain()
