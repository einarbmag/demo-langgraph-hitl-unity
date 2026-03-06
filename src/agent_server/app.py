"""
FastAPI application for the case-management agent server.

Endpoints
---------
POST /api/cases
    Unity submits a new case.  The agent is kicked off, runs until the HITL
    interrupt, and the thread_id is returned so Unity can associate it with
    the case.

POST /copilotkit
    CopilotKit AG-UI SSE endpoint.  The HITL frontend connects here using
    the thread_id obtained from Unity.  When the thread has an active
    interrupt the endpoint immediately replays it so the frontend can render
    the approval UI without requiring additional user input.

GET /copilotkit/health
    Liveness check (added automatically by add_langgraph_fastapi_endpoint).
"""

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, EmailStr

from ag_ui_langgraph import add_langgraph_fastapi_endpoint

from .agui_agent import StateSyncAGUIAgent
from .graph import graph, shared_checkpointer

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Case Management Agent Server")


# ---------------------------------------------------------------------------
# AG-UI endpoint (CopilotKit HITL frontend)
# ---------------------------------------------------------------------------

add_langgraph_fastapi_endpoint(
    app=app,
    agent=StateSyncAGUIAgent(
        name="case_agent",
        description="Reviews support cases and requests human approval before acting.",
        graph=graph,
    ),
    path="/copilotkit",
)


# ---------------------------------------------------------------------------
# Unity intake endpoint
# ---------------------------------------------------------------------------

class CaseIntakeRequest(BaseModel):
    client_email: str
    unity_webhook_url: str
    case_id: Optional[str] = None


class CaseIntakeResponse(BaseModel):
    thread_id: str
    case_id: str


@app.post("/api/cases", response_model=CaseIntakeResponse)
async def intake_case(body: CaseIntakeRequest) -> CaseIntakeResponse:
    """
    Unity submits a new support case.

    The agent is invoked synchronously up to its first interrupt (the HITL
    approval step).  The thread_id is returned so Unity can pass it to the
    HITL frontend via a URL parameter.
    """
    case_id = body.case_id or str(uuid.uuid4())
    thread_id = str(uuid.uuid4())

    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "case_id": case_id,
        "client_email": body.client_email,
        "unity_webhook_url": body.unity_webhook_url,
        "messages": [],
    }

    # ainvoke runs the graph until it hits the interrupt() in
    # request_hitl_approval, then returns.  The checkpoint is saved so the
    # HITL frontend can reconnect and see the pending interrupt.
    await graph.ainvoke(initial_state, config=config)

    return CaseIntakeResponse(thread_id=thread_id, case_id=case_id)
