"""
backend/main.py

Tauron FastAPI server — localhost:8000

Endpoints:
  GET /herd              — risk scores + adjacency matrix for D3.js graph
  GET /explain/{cow_id}  — gradient XAI + LLM alert for a specific cow

Mock mode:
  Set USE_MOCK = True in mock_data.py to serve hardcoded responses.
  Flip to False once tauron_model.pt is trained and graph_utils.py is implemented.
  Emergency rollback: flip USE_MOCK back to True — demo reverts in 30 seconds.

CORS: allow_origins=["*"] is intentional for localhost dev — lock down if deployed.
"""

import io
from datetime import date, datetime
from typing import Any, List

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.mock_data import MOCK_HERD, MOCK_EXPLAIN, USE_MOCK


# ---------------------------------------------------------------------------
# Response models
# Fields marked (new) are additions from multi-disease model — frontend can ignore
# unknown fields (additive changes are backwards-compatible with existing D3.js consumers).
# ---------------------------------------------------------------------------

class CowSummary(BaseModel):
    id: int
    risk_score: float
    status: str                      # "alert" | "watch" | "ok"
    top_feature: str | None          # highest-gradient sensor signal; null for "ok" cows
    dominant_disease: str | None     # (new) "mastitis" | "brd" | "lameness"; null if ok
    all_risks: dict | None           # (new) {disease: score}; null if ok


class HerdResponse(BaseModel):
    cows: list[CowSummary]
    adjacency: list[list[int]]


class ExplainResponse(BaseModel):
    cow_id: int
    risk_score: float
    top_edge: dict                   # {"from": int, "to": int, "weight": float}
    top_feature: str                 # e.g. "milk_yield_kg"
    feature_delta: float             # signed change vs 6-day baseline
    dominant_disease: str | None     # (new) primary disease risk
    all_risks: dict | None           # (new) full disease breakdown
    alert_text: str                  # plain-English farmer alert


class IngestRecord(BaseModel):
    cow_id: int
    date: str
    metric: str
    value: Any                       # float for numeric sensors, str for pen_id


class IngestResponse(BaseModel):
    status: str
    rows_ingested: int
    records: List[IngestRecord]


# ---------------------------------------------------------------------------
# In-memory record store — per-process, resets on restart
# ---------------------------------------------------------------------------

_records: List[dict] = []


# ---------------------------------------------------------------------------
# Normalisation helpers — all return List[{cow_id, date, metric, value}]
# ---------------------------------------------------------------------------

def _normalize_manual(body: dict) -> List[dict]:
    cow_id = int(body["cow_id"])
    dt = str(body.get("date") or date.today().isoformat())
    out = []
    if body.get("milk_yield_kg") is not None:
        out.append({"cow_id": cow_id, "date": dt, "metric": "milk_yield_kg",
                    "value": float(body["milk_yield_kg"])})
    if body.get("pen_id"):
        out.append({"cow_id": cow_id, "date": dt, "metric": "pen_id",
                    "value": str(body["pen_id"])})
    event = body.get("health_event")
    if event and event != "none":
        out.append({"cow_id": cow_id, "date": dt, "metric": "health_event", "value": 1.0})
        out.append({"cow_id": cow_id, "date": dt, "metric": "health_event_type",
                    "value": str(event)})
    return out


def _normalize_webhook(body: dict) -> List[dict]:
    ts = body.get("timestamp") or date.today().isoformat()
    try:
        dt = str(datetime.fromisoformat(str(ts).replace("Z", "+00:00")).date())
    except (ValueError, AttributeError):
        dt = date.today().isoformat()
    return [{"cow_id": int(body["cow_id"]), "date": dt,
             "metric": str(body["metric"]), "value": body["value"]}]


def _normalize_batch(records: list) -> List[dict]:
    out = []
    for r in records:
        out.extend(_normalize_manual(r))
    return out


def _normalize_csv(df: pd.DataFrame) -> List[dict]:
    if "date" not in df.columns:
        df = df.copy()
        df["date"] = date.today().isoformat()
    id_cols = {"cow_id", "date"}
    metric_cols = [c for c in df.columns if c not in id_cols]
    out = []
    for _, row in df.iterrows():
        cow_id = int(row["cow_id"])
        dt = str(row["date"])
        for col in metric_cols:
            val = row[col]
            if pd.notna(val):
                out.append({"cow_id": cow_id, "date": dt, "metric": col, "value": val})
    return out


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Tauron API",
    description=(
        "Early warning system for dairy herd disease detection. "
        "GraphSAGE + GRU model predicting mastitis, BRD, and lameness risk "
        "48 hours ahead. Gradient-based XAI with local Mistral-7B farmer alerts."
    ),
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # intentional for localhost dev — see claude.md
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/herd", response_model=HerdResponse)
async def get_herd():
    """
    Returns risk scores for all cows and the adjacency matrix.

    Each cow entry includes:
    - risk_score: max risk across mastitis, BRD, and lameness [0.0, 1.0]
    - status: "alert" (>0.70), "watch" (0.40–0.70), "ok" (<0.40)
    - dominant_disease: which disease is driving the risk (null if ok)
    - all_risks: individual scores per disease (null if ok)

    Adjacency matrix row/col order matches the cows list order exactly.
    """
    if USE_MOCK:
        return MOCK_HERD

    from backend.graph_utils import build_graph, run_inference
    graph = build_graph()
    return run_inference(graph)


@app.get("/explain/{cow_id}", response_model=ExplainResponse)
async def get_explain(cow_id: int):
    """
    Returns gradient XAI output + LLM-generated plain-English alert for one cow.

    Runs the full explanation pipeline:
    1. Builds contact graph from farm records
    2. Runs TauronGNN forward + backward pass for gradient attribution
    3. Identifies highest-importance sensor signal and contact edge
    4. Generates plain-English alert via local Mistral-7B (Ollama) or Claude API

    The alert text tells the farmer:
    - Which cow (#ID)
    - What disease risk (mastitis / BRD / lameness)
    - What sensor signal triggered it (e.g. milk yield dropped 18%)
    - Which contact cow is most relevant
    - What action to take (isolate / check / monitor)
    """
    if USE_MOCK:
        if cow_id not in MOCK_EXPLAIN:
            raise HTTPException(
                status_code=404,
                detail=f"Cow {cow_id} not found. Available IDs: {list(MOCK_EXPLAIN.keys())}",
            )
        return MOCK_EXPLAIN[cow_id]

    from backend.xai_bridge import explain_cow
    try:
        return await explain_cow(cow_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/ingest")
async def get_ingest():
    """Return all ingested records (in-memory store, resets on restart)."""
    return {"status": "ok", "rows_ingested": len(_records), "records": _records}


@app.post("/api/ingest", response_model=IngestResponse)
async def post_ingest(request: Request):
    """
    Ingest farm sensor data. Three accepted formats:

    1. CSV upload (multipart/form-data, field name `file`)
    2. Batch JSON: {"records": [{cow_id, date?, milk_yield_kg?, pen_id?, health_event?}]}
    3. Webhook:   {"cow_id": 47, "metric": "milk_yield_kg", "value": 24.5, "timestamp": "..."}

    All normalize to: [{cow_id, date, metric, value}]
    """
    ct = request.headers.get("content-type", "")

    if "multipart/form-data" in ct:
        form = await request.form()
        file = form.get("file")
        if file is None or not hasattr(file, "read"):
            raise HTTPException(400, "multipart request must include a `file` field")
        content = await file.read()
        try:
            df = pd.read_csv(io.StringIO(content.decode()))
        except Exception as exc:
            raise HTTPException(422, f"CSV parse error: {exc}")
        if "cow_id" not in df.columns:
            raise HTTPException(422, "CSV must have a `cow_id` column")
        records = _normalize_csv(df)
    else:
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(400, "body must be valid JSON or multipart/form-data")
        if not isinstance(body, dict) or "cow_id" not in body:
            raise HTTPException(422, "JSON body must contain `cow_id`")
        if "records" in body:
            records = _normalize_batch(body["records"])
        elif "metric" in body:
            records = _normalize_webhook(body)
        else:
            records = _normalize_manual(body)

    if not records:
        raise HTTPException(422, "no records could be parsed from the provided data")

    _records.extend(records)
    return {"status": "ok", "rows_ingested": len(records), "records": records}
