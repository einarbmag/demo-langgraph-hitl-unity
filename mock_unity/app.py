"""
Mock Unity case management server.

Simulates the two things Unity actually does in the real world:

1. POST /api/cases → agent server   (done by tests; not here)
2. Receives webhook callbacks from the agent when a case is resolved.

Endpoints
---------
POST /webhook/case-update
    The agent's finalize_case node POSTs the case outcome here.

GET  /webhook/case-updates
    Returns all received case updates so tests can assert on them.

DELETE /webhook/case-updates
    Clears the in-memory store (called between tests for isolation).
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Mock Unity Server")

# In-memory store of received webhook payloads (reset between tests).
_received_updates: list[dict[str, Any]] = []


class CaseUpdatePayload(BaseModel):
    case_id: str
    client_email: str
    recommended_action: str
    hitl_decision: str
    hitl_notes: str
    status: str


@app.post("/webhook/case-update", status_code=200)
async def receive_case_update(payload: CaseUpdatePayload) -> dict:
    """Record the case update that the agent posts when a case is finalised."""
    _received_updates.append(payload.model_dump())
    return {"received": True}


@app.get("/webhook/case-updates")
async def get_case_updates() -> list[dict]:
    """Return all recorded case updates (used by tests)."""
    return list(_received_updates)


@app.delete("/webhook/case-updates")
async def clear_case_updates() -> dict:
    """Reset the store between tests."""
    _received_updates.clear()
    return {"cleared": True}
