"""
Layer 3 – AG-UI state-sync tests.

These tests verify the critical requirement:
  "The HITL UI should sync the agent state so the user immediately sees the
   agent history and the HITL request."

When the frontend connects to POST /copilotkit with a thread_id that has an
active interrupt, ag_ui_langgraph's prepare_stream detects the pending
interrupt and immediately re-emits it – no re-execution of the graph.

We drive the AG-UI endpoint directly using a raw HTTP client, parse the
Server-Sent Events (SSE) stream, and assert on the event sequence.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import pytest
import httpx
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from agent_server.graph import build_graph
from agent_server.app import app as agent_app


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

def parse_sse_events(raw: str) -> list[dict[str, Any]]:
    """
    Parse a raw SSE response body into a list of event dicts.

    Each SSE event looks like::

        event: <type>\\ndata: <json>\\n\\n
    """
    events = []
    current: dict[str, str] = {}
    for line in raw.splitlines():
        if line.startswith("event:"):
            current["event"] = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_str = line[len("data:"):].strip()
            try:
                current["data"] = json.loads(data_str)
            except json.JSONDecodeError:
                current["data"] = data_str
        elif line == "" and current:
            events.append(current)
            current = {}
    if current:
        events.append(current)
    return events


def agui_request(thread_id: str, forwarded_props: dict | None = None) -> dict:
    """Build a minimal valid RunAgentInput payload."""
    return {
        "thread_id": thread_id,
        "run_id": str(uuid.uuid4()),
        "state": {},
        "messages": [],
        "tools": [],
        "context": [],
        "forwarded_props": forwarded_props or {},
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def isolated_graph():
    """Graph with its own checkpointer, independent of the shared app graph."""
    return build_graph(checkpointer=MemorySaver())


@pytest.fixture()
async def patched_agent_client(isolated_graph):
    """Agent HTTP client backed by an isolated graph."""
    import agent_server.app as mod

    original = mod.graph
    mod.graph = isolated_graph

    # Also patch the LangGraphAGUIAgent that was already registered on the app
    from copilotkit import LangGraphAGUIAgent
    from ag_ui_langgraph.agent import LangGraphAgent

    for route in agent_app.routes:
        endpoint = getattr(route, "endpoint", None)
        if endpoint is None:
            continue
        # The closure inside add_langgraph_fastapi_endpoint captures `agent`
        closure_vars = getattr(endpoint, "__closure__", None) or []
        for cell in closure_vars:
            try:
                obj = cell.cell_contents
            except ValueError:
                continue
            if isinstance(obj, (LangGraphAGUIAgent, LangGraphAgent)):
                obj.graph = isolated_graph
                break

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=agent_app), base_url="http://test"
    ) as client:
        yield client, isolated_graph

    mod.graph = original


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAGUIStateSync:
    @pytest.mark.asyncio
    async def test_initial_run_emits_run_started_and_finished(self, patched_agent_client):
        """A fresh run (no prior state) should emit RUN_STARTED … RUN_FINISHED."""
        client, graph = patched_agent_client
        thread_id = str(uuid.uuid4())

        # First, intake the case so the graph has an interrupt checkpoint
        await graph.ainvoke(
            {
                "case_id": "CASE-SYNC-1",
                "client_email": "sync@billing.example.com",
                "unity_webhook_url": "http://unity/webhook/case-update",
                "messages": [],
            },
            config={"configurable": {"thread_id": thread_id}},
        )

        # Now connect via AG-UI as the frontend would
        async with client.stream(
            "POST",
            "/copilotkit",
            json=agui_request(thread_id),
            headers={"Accept": "text/event-stream"},
        ) as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers.get("content-type", "")
            raw = await resp.aread()

        events = parse_sse_events(raw.decode())
        event_types = [e.get("data", {}).get("type") if isinstance(e.get("data"), dict) else e.get("event") for e in events]

        # Should contain RUN_STARTED and RUN_FINISHED
        assert any("RUN_STARTED" in str(t) for t in event_types), f"No RUN_STARTED in {event_types}"
        assert any("RUN_FINISHED" in str(t) for t in event_types), f"No RUN_FINISHED in {event_types}"

    @pytest.mark.asyncio
    async def test_reconnecting_to_interrupted_thread_replays_interrupt(
        self, patched_agent_client
    ):
        """
        KEY TEST: When the HITL frontend connects to a thread that is paused
        at an interrupt, the AG-UI endpoint must immediately emit the interrupt
        event – without re-running the graph.

        This is the 'state sync' behaviour that lets the user see the pending
        approval request as soon as they open the case in the UI.
        """
        client, graph = patched_agent_client
        thread_id = str(uuid.uuid4())

        # Step 1: Unity submits case → graph runs to interrupt checkpoint
        await graph.ainvoke(
            {
                "case_id": "CASE-SYNC-2",
                "client_email": "sync2@vip.example.com",
                "unity_webhook_url": "http://unity/webhook/case-update",
                "messages": [],
            },
            config={"configurable": {"thread_id": thread_id}},
        )

        # Step 2: HITL frontend connects (simulated by calling the AG-UI endpoint)
        async with client.stream(
            "POST",
            "/copilotkit",
            json=agui_request(thread_id),
            headers={"Accept": "text/event-stream"},
        ) as resp:
            raw = await resp.aread()

        events = parse_sse_events(raw.decode())

        # ag_ui_langgraph emits {type: "CUSTOM", name: "on_interrupt", value: "<json>"}
        # Note: the event name is lowercase "on_interrupt" not "OnInterrupt"
        interrupt_events = [
            e for e in events
            if (
                isinstance(e.get("data"), dict)
                and e["data"].get("type") == "CUSTOM"
                and "interrupt" in str(e["data"].get("name", "")).lower()
            )
        ]
        assert len(interrupt_events) >= 1, (
            f"Expected at least one on_interrupt CUSTOM event; got events:\n"
            + "\n".join(str(e) for e in events)
        )

    @pytest.mark.asyncio
    async def test_interrupt_event_contains_decision_context(self, patched_agent_client):
        """
        The interrupt payload contains the decision-specific context only:
        the question text, the case_id, and the recommended_action.

        Full state (case_summary, client_email, …) is NOT duplicated here –
        it arrives via STATE_SNAPSHOT (tested separately below), where it
        populates useCoAgent automatically.
        """
        client, graph = patched_agent_client
        thread_id = str(uuid.uuid4())

        await graph.ainvoke(
            {
                "case_id": "CASE-SYNC-3",
                "client_email": "sync3@fraud.example.com",
                "unity_webhook_url": "http://unity/webhook/case-update",
                "messages": [],
            },
            config={"configurable": {"thread_id": thread_id}},
        )

        async with client.stream(
            "POST",
            "/copilotkit",
            json=agui_request(thread_id),
            headers={"Accept": "text/event-stream"},
        ) as resp:
            raw = await resp.aread()

        events = parse_sse_events(raw.decode())

        interrupt_value = None
        for e in events:
            data = e.get("data", {})
            if isinstance(data, dict) and data.get("type") == "CUSTOM" and "interrupt" in str(data.get("name", "")).lower():
                raw_val = data.get("value")
                interrupt_value = json.loads(raw_val) if isinstance(raw_val, str) else raw_val
                break

        assert interrupt_value is not None, "No interrupt event found"
        # Decision-specific fields ARE present
        assert interrupt_value.get("case_id") == "CASE-SYNC-3"
        assert interrupt_value.get("recommended_action") == "flag_for_fraud_review"
        assert interrupt_value.get("question") is not None
        # Full state fields are NOT duplicated here (they're in STATE_SNAPSHOT)
        assert "case_summary" not in interrupt_value
        assert "client_email" not in interrupt_value

    @pytest.mark.asyncio
    async def test_state_snapshot_emitted_on_reconnect(self, patched_agent_client):
        """
        KEY TEST: StateSyncAGUIAgent injects STATE_SNAPSHOT into the reconnect
        fast-path, so useCoAgent is populated automatically.

        The frontend's useCoAgent.state will contain case_summary, client_email,
        recommended_action etc. without any manual copying into the interrupt payload.
        """
        client, graph = patched_agent_client
        thread_id = str(uuid.uuid4())

        await graph.ainvoke(
            {
                "case_id": "CASE-SYNC-4",
                "client_email": "sync4@billing.example.com",
                "unity_webhook_url": "http://unity/webhook/case-update",
                "messages": [],
            },
            config={"configurable": {"thread_id": thread_id}},
        )

        async with client.stream(
            "POST",
            "/copilotkit",
            json=agui_request(thread_id),
            headers={"Accept": "text/event-stream"},
        ) as resp:
            raw = await resp.aread()

        events = parse_sse_events(raw.decode())

        event_types = [
            e.get("data", {}).get("type")
            for e in events
            if isinstance(e.get("data"), dict)
        ]

        # StateSyncAGUIAgent adds these to the fast-path
        assert "STATE_SNAPSHOT" in event_types, (
            "STATE_SNAPSHOT must be emitted on reconnect so useCoAgent is populated\n"
            + str(event_types)
        )
        assert "MESSAGES_SNAPSHOT" in event_types, (
            "MESSAGES_SNAPSHOT must be emitted on reconnect for message history\n"
            + str(event_types)
        )

        # Find the STATE_SNAPSHOT and verify it contains the full case state
        snapshot_event = next(
            e for e in events
            if isinstance(e.get("data"), dict)
            and e["data"].get("type") == "STATE_SNAPSHOT"
        )
        snapshot = snapshot_event["data"].get("snapshot", {})
        assert snapshot.get("case_id") == "CASE-SYNC-4"
        assert snapshot.get("case_summary") is not None
        assert "sync4@billing" in snapshot.get("case_summary", "")
        assert snapshot.get("recommended_action") is not None

        # Event order: RunStarted → STATE_SNAPSHOT → MESSAGES_SNAPSHOT → on_interrupt → RunFinished
        non_raw_types = [t for t in event_types if t not in (None,)]
        state_idx = non_raw_types.index("STATE_SNAPSHOT")
        interrupt_idx = next(
            i for i, e in enumerate(events)
            if isinstance(e.get("data"), dict)
            and e["data"].get("type") == "CUSTOM"
            and "interrupt" in str(e["data"].get("name", "")).lower()
        )
        # STATE_SNAPSHOT event index must be before on_interrupt event index
        state_event_pos = next(
            i for i, e in enumerate(events)
            if isinstance(e.get("data"), dict) and e["data"].get("type") == "STATE_SNAPSHOT"
        )
        assert state_event_pos < interrupt_idx, (
            "STATE_SNAPSHOT must come before on_interrupt"
        )

    @pytest.mark.asyncio
    async def test_resume_via_agui_completes_graph(self, patched_agent_client, respx_mock):
        """
        The frontend resolves the interrupt by sending forwarded_props with a
        resume command.  The graph should then run finalize_case and reach END.
        """
        client, graph = patched_agent_client
        thread_id = str(uuid.uuid4())

        # Step 1: Initial run → interrupt
        await graph.ainvoke(
            {
                "case_id": "CASE-SYNC-5",
                "client_email": "sync5@vip.example.com",
                "unity_webhook_url": "http://unity/webhook/case-update",
                "messages": [],
            },
            config={"configurable": {"thread_id": thread_id}},
        )

        # Step 2: Mock Unity webhook
        respx_mock.post("http://unity/webhook/case-update").respond(
            200, json={"received": True}
        )

        # Step 3: Frontend sends resume command via AG-UI
        resume_props = {"command": {"resume": {"decision": "approve", "notes": "Approved via UI"}}}
        async with client.stream(
            "POST",
            "/copilotkit",
            json=agui_request(thread_id, forwarded_props=resume_props),
            headers={"Accept": "text/event-stream"},
        ) as resp:
            raw = await resp.aread()

        events = parse_sse_events(raw.decode())
        assert any("RUN_FINISHED" in str(e) for e in events), "Expected RUN_FINISHED after resume"

        # Step 4: Graph should be at END
        snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
        assert snapshot.next == ()
        assert snapshot.values["status"] == "complete"
        assert snapshot.values["hitl_decision"] == "approve"


# ---------------------------------------------------------------------------
# Completed-thread history viewing
# ---------------------------------------------------------------------------

class TestCompletedThread:
    """
    Verifies that a user can open the HITL UI for a case that has *already*
    been resolved (graph at END, no pending interrupt) and see the final
    state and history without triggering a new graph run.

    Use-case: an auditor or supervisor opens a case after it was approved or
    rejected to review the decision.
    """

    @pytest.mark.asyncio
    async def test_connecting_to_completed_thread_shows_final_state(
        self, patched_agent_client, respx_mock
    ):
        """
        The HITL UI connects to a thread whose graph has already reached END.
        StateSyncAGUIAgent must emit the final checkpointed state directly
        (STATE_SNAPSHOT + MESSAGES_SNAPSHOT) instead of re-running the graph.

        Without the fix:
        - ag_ui_langgraph re-invokes graph.astream_events(input=state)
        - LangGraph restarts the graph from START
        - process_case + request_hitl_approval run again → new interrupt
        - The completed case is overwritten with a fresh one

        With the fix:
        - STATE_SNAPSHOT shows status="complete" and the approve/reject decision
        - RUN_FINISHED is emitted (no dangling stream)
        - No on_interrupt CUSTOM event is present
        """
        client, graph = patched_agent_client
        thread_id = str(uuid.uuid4())

        # ── Step 1: Submit case → graph pauses at interrupt ──────────────────
        await graph.ainvoke(
            {
                "case_id": "CASE-DONE-1",
                "client_email": "done@billing.example.com",
                "unity_webhook_url": "http://unity/webhook/case-update",
                "messages": [],
            },
            config={"configurable": {"thread_id": thread_id}},
        )

        # ── Step 2: Approve → graph runs finalize_case → END ─────────────────
        respx_mock.post("http://unity/webhook/case-update").respond(
            200, json={"received": True}
        )
        from langgraph.types import Command
        await graph.ainvoke(
            Command(resume={"decision": "approve", "notes": "Looks clean"}),
            config={"configurable": {"thread_id": thread_id}},
        )

        completed = await graph.aget_state({"configurable": {"thread_id": thread_id}})
        assert completed.next == (), "Graph must be at END before testing reconnect"
        assert completed.values["status"] == "complete"

        # ── Step 3: HITL UI opens the completed case ──────────────────────────
        async with client.stream(
            "POST",
            "/copilotkit",
            json=agui_request(thread_id),
            headers={"Accept": "text/event-stream"},
        ) as resp:
            raw = await resp.aread()

        events = parse_sse_events(raw.decode())
        event_types = [
            e.get("data", {}).get("type")
            for e in events
            if isinstance(e.get("data"), dict)
        ]

        # ── Lifecycle events ──────────────────────────────────────────────────
        assert any("RUN_STARTED" in str(t) for t in event_types), (
            f"Missing RUN_STARTED in {event_types}"
        )
        assert any("RUN_FINISHED" in str(t) for t in event_types), (
            f"Missing RUN_FINISHED in {event_types}"
        )

        # ── STATE_SNAPSHOT must reflect the completed state ───────────────────
        assert "STATE_SNAPSHOT" in event_types, (
            "StateSyncAGUIAgent must emit STATE_SNAPSHOT for completed threads "
            "so useCoAgent shows the final decision\n"
            + str(event_types)
        )

        snapshot_event = next(
            e for e in events
            if isinstance(e.get("data"), dict)
            and e["data"].get("type") == "STATE_SNAPSHOT"
        )
        snap = snapshot_event["data"].get("snapshot", {})
        assert snap.get("status") == "complete", (
            f"Expected status='complete' in STATE_SNAPSHOT, got: {snap.get('status')}"
        )
        assert snap.get("hitl_decision") == "approve", (
            f"Expected hitl_decision='approve', got: {snap.get('hitl_decision')}"
        )
        assert snap.get("case_id") == "CASE-DONE-1"
        assert "done@billing" in snap.get("case_summary", ""), (
            "case_summary should contain the client email"
        )

        # ── MESSAGES_SNAPSHOT for history ─────────────────────────────────────
        assert "MESSAGES_SNAPSHOT" in event_types, (
            f"Missing MESSAGES_SNAPSHOT in {event_types}"
        )

        # ── No interrupt event – the graph is done ────────────────────────────
        interrupt_events = [
            e for e in events
            if (
                isinstance(e.get("data"), dict)
                and e["data"].get("type") == "CUSTOM"
                and "interrupt" in str(e["data"].get("name", "")).lower()
            )
        ]
        assert len(interrupt_events) == 0, (
            "Completed thread must NOT emit on_interrupt – "
            "the case is already resolved\n"
            + "\n".join(str(e) for e in events)
        )

    @pytest.mark.asyncio
    async def test_completed_thread_does_not_restart_graph(
        self, patched_agent_client, respx_mock
    ):
        """
        Guard: the graph must NOT be re-invoked when connecting to a completed
        thread.  After the HITL UI connects, the graph state should remain
        unchanged (status still "complete", same hitl_decision).

        This catches the regression where ag_ui_langgraph would restart the
        graph from START, overwriting the completed case with a fresh run.
        """
        client, graph = patched_agent_client
        thread_id = str(uuid.uuid4())

        # Submit + approve
        await graph.ainvoke(
            {
                "case_id": "CASE-DONE-2",
                "client_email": "nochange@vip.example.com",
                "unity_webhook_url": "http://unity/webhook/case-update",
                "messages": [],
            },
            config={"configurable": {"thread_id": thread_id}},
        )
        respx_mock.post("http://unity/webhook/case-update").respond(
            200, json={"received": True}
        )
        from langgraph.types import Command
        await graph.ainvoke(
            Command(resume={"decision": "reject", "notes": "Declined"}),
            config={"configurable": {"thread_id": thread_id}},
        )

        before = await graph.aget_state({"configurable": {"thread_id": thread_id}})
        assert before.next == ()

        # Open HITL UI (should only read, not mutate)
        async with client.stream(
            "POST",
            "/copilotkit",
            json=agui_request(thread_id),
            headers={"Accept": "text/event-stream"},
        ) as resp:
            await resp.aread()

        after = await graph.aget_state({"configurable": {"thread_id": thread_id}})

        # Graph state must be identical before and after the UI connects
        assert after.next == (), "Graph must still be at END after HITL UI connects"
        assert after.values["status"] == "complete", (
            "status must remain 'complete' – graph was re-run if this fails"
        )
        assert after.values["hitl_decision"] == "reject", (
            "hitl_decision must remain 'reject' – graph was re-run if this fails"
        )
        assert after.values["case_id"] == "CASE-DONE-2", (
            "case_id must not change – graph was re-run if this fails"
        )
