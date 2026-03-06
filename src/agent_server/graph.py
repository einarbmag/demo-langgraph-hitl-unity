"""
LangGraph case-management agent.

Flow:
  START → process_case → request_hitl_approval (interrupt) → finalize_case → END

The graph is deterministic – no LLM calls.  It is compiled once (module level)
with an in-memory checkpointer so that the same checkpointer is shared between
the Unity intake endpoint and the CopilotKit AG-UI endpoint.
"""

from __future__ import annotations

import hashlib
from typing import Optional

import httpx
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from copilotkit import CopilotKitState

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class CaseState(CopilotKitState):
    """State for the case-management agent.

    Inherits ``messages`` (with add_messages reducer) and ``copilotkit``
    from CopilotKitState / MessagesState.
    """

    # Provided by Unity on intake
    case_id: str
    client_email: str
    unity_webhook_url: str

    # Populated by process_case
    case_summary: Optional[str]
    recommended_action: Optional[str]

    # Populated after HITL approval
    hitl_decision: Optional[str]   # "approve" | "reject"
    hitl_notes: Optional[str]

    # Lifecycle status
    status: Optional[str]          # "processing" | "awaiting_approval" | "approved" | "rejected" | "complete"


# ---------------------------------------------------------------------------
# Node: process_case
# ---------------------------------------------------------------------------

_ACTION_MAP: dict[str, str] = {
    "vip": "escalate_to_senior_agent",
    "fraud": "flag_for_fraud_review",
    "billing": "issue_billing_credit",
}

def _derive_action(client_email: str) -> str:
    """Deterministically pick a recommended action based on the email domain."""
    domain = client_email.split("@")[-1].split(".")[0].lower()
    for keyword, action in _ACTION_MAP.items():
        if keyword in domain or keyword in client_email.lower():
            return action
    # Stable fallback based on hash of email
    digest = int(hashlib.md5(client_email.encode()).hexdigest(), 16) % len(_ACTION_MAP)
    return list(_ACTION_MAP.values())[digest]


async def process_case(state: CaseState, config: RunnableConfig) -> dict:
    """Process incoming case data and produce a case summary + recommended action."""
    client_email = state["client_email"]
    case_id = state["case_id"]

    case_summary = (
        f"Support case {case_id} submitted by {client_email}. "
        f"Client account requires immediate attention."
    )
    recommended_action = _derive_action(client_email)

    return {
        "case_summary": case_summary,
        "recommended_action": recommended_action,
        "status": "awaiting_approval",
    }


# ---------------------------------------------------------------------------
# Node: request_hitl_approval
# ---------------------------------------------------------------------------

async def request_hitl_approval(state: CaseState, config: RunnableConfig) -> dict:
    """Pause execution and request a human decision via CopilotKit HITL.

    The interrupt payload carries only what is *decision-specific*: the
    question text and the action being proposed.

    The rest of the case context (case_summary, client_email, status, …) is
    already in the graph state.  Because StateSyncAGUIAgent injects a
    STATE_SNAPSHOT event on reconnect, the frontend's useCoAgent hook receives
    the full state automatically – no manual duplication required.
    """
    decision_payload: dict = interrupt(
        {
            "question": "Please review the case and approve or reject the recommended action.",
            "case_id": state["case_id"],
            "recommended_action": state["recommended_action"],
        }
    )

    # decision_payload is whatever the HITL frontend resolves with:
    # {"decision": "approve"|"reject", "notes": "<optional text>"}
    decision = decision_payload.get("decision", "reject")
    notes = decision_payload.get("notes", "")

    return {
        "hitl_decision": decision,
        "hitl_notes": notes,
        "status": decision,  # "approve" | "reject"
    }


# ---------------------------------------------------------------------------
# Node: finalize_case
# ---------------------------------------------------------------------------

async def finalize_case(state: CaseState, config: RunnableConfig) -> dict:
    """Post the outcome back to Unity and mark the case complete."""
    payload = {
        "case_id": state["case_id"],
        "client_email": state["client_email"],
        "recommended_action": state["recommended_action"],
        "hitl_decision": state["hitl_decision"],
        "hitl_notes": state["hitl_notes"],
        "status": "complete",
    }

    webhook_url = state.get("unity_webhook_url", "")
    if webhook_url:
        async with httpx.AsyncClient() as client:
            await client.post(webhook_url, json=payload, timeout=10.0)

    return {"status": "complete"}


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph(checkpointer: MemorySaver | None = None) -> object:
    """Build and compile the case-management graph.

    Parameters
    ----------
    checkpointer:
        Pass an explicit checkpointer (useful in tests).  Defaults to a new
        ``MemorySaver`` when *None*.
    """
    if checkpointer is None:
        checkpointer = MemorySaver()

    builder = StateGraph(CaseState)
    builder.add_node("process_case", process_case)
    builder.add_node("request_hitl_approval", request_hitl_approval)
    builder.add_node("finalize_case", finalize_case)

    builder.add_edge(START, "process_case")
    builder.add_edge("process_case", "request_hitl_approval")
    builder.add_edge("request_hitl_approval", "finalize_case")
    builder.add_edge("finalize_case", END)

    return builder.compile(checkpointer=checkpointer)


# Shared checkpointer – used by both the intake REST endpoint and the AG-UI
# endpoint so they operate on the same persisted thread state.
shared_checkpointer = MemorySaver()
graph = build_graph(checkpointer=shared_checkpointer)
