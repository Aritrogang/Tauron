"""
backend/main.py

Tauron FastAPI server — localhost:8000

Endpoints:
  GET /herd              — risk scores + adjacency matrix for D3.js graph
  GET /explain/{cow_id}  — GNNExplainer + LLM alert for a specific cow

Mock mode:
  Set USE_MOCK = True in mock_data.py to serve hardcoded responses.
  This lets the Frontend team build D3.js components before ML is integrated.
  Flip to False after ML team delivers tauron_model.pt + graph_utils.py stubs.

CORS: allow_origins=["*"] is intentional for localhost dev — lock down if deployed.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.mock_data import MOCK_HERD, MOCK_EXPLAIN, USE_MOCK


# ---------------------------------------------------------------------------
# Response models (define the frozen JSON contract for the frontend team)
# ---------------------------------------------------------------------------

class CowSummary(BaseModel):
    id: int
    risk_score: float
    status: str            # "alert" | "watch" | "ok"
    top_feature: str | None


class HerdResponse(BaseModel):
    cows: list[CowSummary]
    adjacency: list[list[int]]


class ExplainResponse(BaseModel):
    cow_id: int
    risk_score: float
    top_edge: dict         # {"from": int, "to": int, "weight": float}
                           # Note: "from" is a reserved Python keyword — kept as dict
    top_feature: str
    feature_delta: float
    alert_text: str


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Tauron API",
    description=(
        "Early warning system for dairy herd disease detection. "
        "GraphSAGE + GRU model with GNNExplainer XAI and local Mistral-7B alerts."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # intentional for localhost dev — see claude.md
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/herd", response_model=HerdResponse)
async def get_herd():
    """
    Returns risk scores for all cows and the adjacency matrix.
    The adjacency matrix row/col order matches the cows list order exactly.
    """
    if USE_MOCK:
        return MOCK_HERD

    # Real path — lazy import so server starts without PyTorch installed
    from backend.graph_utils import build_graph, run_inference
    graph = build_graph()
    return run_inference(graph)


@app.get("/explain/{cow_id}", response_model=ExplainResponse)
async def get_explain(cow_id: int):
    """
    Returns GNNExplainer output + Mistral-generated alert for a specific cow.
    Only runs full XAI pipeline for cows with risk_score > 0.70.
    """
    if USE_MOCK:
        if cow_id not in MOCK_EXPLAIN:
            raise HTTPException(
                status_code=404,
                detail=f"Cow {cow_id} not found. Available IDs: {list(MOCK_EXPLAIN.keys())}"
            )
        return MOCK_EXPLAIN[cow_id]

    # Real path — lazy import so server starts without PyTorch/Ollama running
    from backend.xai_bridge import explain_cow
    try:
        return await explain_cow(cow_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
