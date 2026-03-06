"""
Layer 4 – Full end-to-end flow tests.

Exercises the complete sequence described in the requirements:

  1. Unity POST /api/cases  →  thread_id returned
  2. Agent processes case, hits HITL interrupt checkpoint
  3. HITL frontend connects via AG-UI with thread_id  →  interrupt replayed
  4. User resolves interrupt (approve / reject)
  5. Agent finalises case, POSTs result to Unity webhook
  6. Unity webhook receives the correct payload

Both the agent server and mock Unity server run in-process using ASGI
transport – no real network or server processes required.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import httpx
import pytest
import respx

from agent_server.graph import build_graph
from agent_server.app import app as agent_app
from mock_unity.app import app as unity_app


# ---------------------------------------------------------------------------
# Helpers (re-used from test_agui_sync)
# ---------------------------------------------------------------------------

def parse_sse_events(raw: str) -> list[dict[str, Any]]:
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
# Shared fixture: both clients + isolated graph
# ---------------------------------------------------------------------------

@pytest.fixture()
async def clients():
    """
    Yields (agent_client, unity_client, isolated_graph).

    The agent app is patched to use a fresh isolated graph with its own
    MemorySaver, so tests don't interfere with each other.
    """
    from langgraph.checkpoint.memory import MemorySaver

    isolated_graph = build_graph(checkpointer=MemorySaver())

    import agent_server.app as mod
    from ag_ui_langgraph.agent import LangGraphAgent
    from copilotkit import LangGraphAGUIAgent

    original_graph = mod.graph
    mod.graph = isolated_graph

    # Patch the registered AG-UI agent's graph reference
    for route in agent_app.routes:
        endpoint = getattr(route, "endpoint", None)
        if endpoint is None:
            continue
        closure_vars = getattr(endpoint, "__closure__", None) or []
        for cell in closure_vars:
            try:
                obj = cell.cell_contents
            except ValueError:
                continue
            if isinstance(obj, (LangGraphAGUIAgent, LangGraphAgent)):
                obj.graph = isolated_graph
                break

    async with (
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=agent_app), base_url="http://agent"
        ) as agent_client,
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=unity_app), base_url="http://unity"
        ) as unity_client,
    ):
        # Clear any leftover webhook payloads
        await unity_client.delete("/webhook/case-updates")
        yield agent_client, unity_client, isolated_graph

    mod.graph = original_graph


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFullFlow:
    @pytest.mark.asyncio
    async def test_step1_unity_submits_case_gets_thread_id(self, clients):
        """Step 1: Unity POSTs a case and gets a thread_id back."""
        agent_client, unity_client, _ = clients

        resp = await agent_client.post(
            "/api/cases",
            json={
                "client_email": "eve@billing.example.com",
                "unity_webhook_url": "http://unity/webhook/case-update",
                "case_id": "CASE-E2E-01",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "thread_id" in body
        assert body["case_id"] == "CASE-E2E-01"

    @pytest.mark.asyncio
    async def test_step2_agent_awaiting_hitl_after_intake(self, clients):
        """Step 2: After intake the graph is paused at the HITL interrupt."""
        agent_client, _, isolated_graph = clients

        resp = await agent_client.post(
            "/api/cases",
            json={
                "client_email": "eve@billing.example.com",
                "unity_webhook_url": "http://unity/webhook/case-update",
            },
        )
        thread_id = resp.json()["thread_id"]

        snapshot = await isolated_graph.aget_state(
            {"configurable": {"thread_id": thread_id}}
        )
        assert "request_hitl_approval" in snapshot.next
        assert len(snapshot.tasks[0].interrupts) == 1

    @pytest.mark.asyncio
    async def test_step3_hitl_frontend_sees_interrupt_on_connect(self, clients):
        """
        Step 3: The HITL frontend connects with thread_id from URL parameter
        and immediately receives the interrupt event (state sync).
        """
        agent_client, _, isolated_graph = clients

        resp = await agent_client.post(
            "/api/cases",
            json={
                "client_email": "frank@vip.example.com",
                "unity_webhook_url": "http://unity/webhook/case-update",
                "case_id": "CASE-E2E-02",
            },
        )
        thread_id = resp.json()["thread_id"]

        # HITL frontend connects via AG-UI
        async with agent_client.stream(
            "POST",
            "/copilotkit",
            json=agui_request(thread_id),
            headers={"Accept": "text/event-stream"},
        ) as sse_resp:
            raw = await sse_resp.aread()

        events = parse_sse_events(raw.decode())
        # ag_ui_langgraph emits {type: "CUSTOM", name: "on_interrupt", ...}
        interrupt_events = [
            e for e in events
            if (
                isinstance(e.get("data"), dict)
                and e["data"].get("type") == "CUSTOM"
                and "interrupt" in str(e["data"].get("name", "")).lower()
            )
        ]
        assert len(interrupt_events) >= 1, (
            "HITL frontend should receive the pending interrupt on connect\n"
            + "\n".join(str(e) for e in events)
        )

    @pytest.mark.asyncio
    async def test_step3_state_snapshot_contains_case_summary(self, clients):
        """
        Step 3 (continued): StateSyncAGUIAgent injects STATE_SNAPSHOT into the
        reconnect fast-path.  The frontend's useCoAgent.state is therefore
        populated with the full case context (including case_summary) without
        any manual copying into the interrupt payload.
        """
        agent_client, _, isolated_graph = clients

        resp = await agent_client.post(
            "/api/cases",
            json={
                "client_email": "grace@billing.example.com",
                "unity_webhook_url": "http://unity/webhook/case-update",
                "case_id": "CASE-E2E-03",
            },
        )
        thread_id = resp.json()["thread_id"]

        async with agent_client.stream(
            "POST",
            "/copilotkit",
            json=agui_request(thread_id),
            headers={"Accept": "text/event-stream"},
        ) as sse_resp:
            raw = await sse_resp.aread()

        events = parse_sse_events(raw.decode())
        event_types = [
            e.get("data", {}).get("type")
            for e in events
            if isinstance(e.get("data"), dict)
        ]

        # STATE_SNAPSHOT must be present (injected by StateSyncAGUIAgent)
        assert "STATE_SNAPSHOT" in event_types, (
            "Expected STATE_SNAPSHOT on reconnect – is StateSyncAGUIAgent being used?\n"
            + str(event_types)
        )

        # The snapshot should carry the full case state
        snapshot_event = next(
            e for e in events
            if isinstance(e.get("data"), dict)
            and e["data"].get("type") == "STATE_SNAPSHOT"
        )
        snapshot = snapshot_event["data"].get("snapshot", {})
        assert snapshot.get("case_id") == "CASE-E2E-03"
        assert "grace@billing" in snapshot.get("case_summary", "")

    @pytest.mark.asyncio
    async def test_step4_user_approves_via_agui_resume(self, clients, respx_mock):
        """
        Step 4: The user approves the case via the HITL UI.
        The frontend sends a resume command to the AG-UI endpoint.
        """
        agent_client, _, isolated_graph = clients

        resp = await agent_client.post(
            "/api/cases",
            json={
                "client_email": "henry@billing.example.com",
                "unity_webhook_url": "http://unity/webhook/case-update",
            },
        )
        thread_id = resp.json()["thread_id"]

        # Mock the Unity webhook
        respx_mock.post("http://unity/webhook/case-update").respond(
            200, json={"received": True}
        )

        # User approves via AG-UI
        resume_props = {"command": {"resume": {"decision": "approve", "notes": "LGTM"}}}
        async with agent_client.stream(
            "POST",
            "/copilotkit",
            json=agui_request(thread_id, forwarded_props=resume_props),
            headers={"Accept": "text/event-stream"},
        ) as sse_resp:
            raw = await sse_resp.aread()

        events = parse_sse_events(raw.decode())
        assert any("RUN_FINISHED" in str(e) for e in events)

        snapshot = await isolated_graph.aget_state(
            {"configurable": {"thread_id": thread_id}}
        )
        assert snapshot.values["hitl_decision"] == "approve"
        assert snapshot.values["status"] == "complete"

    @pytest.mark.asyncio
    async def test_step5_agent_posts_to_unity_webhook(self, clients, respx_mock):
        """
        Step 5: After approval, finalize_case POSTs the case outcome to the
        Unity webhook URL that was in the original case payload.
        """
        agent_client, _, _ = clients

        webhook_route = respx_mock.post("http://unity/webhook/case-update").respond(
            200, json={"received": True}
        )

        resp = await agent_client.post(
            "/api/cases",
            json={
                "client_email": "iris@vip.example.com",
                "unity_webhook_url": "http://unity/webhook/case-update",
                "case_id": "CASE-E2E-WEBHOOK",
            },
        )
        thread_id = resp.json()["thread_id"]

        # Approve
        resume_props = {"command": {"resume": {"decision": "approve", "notes": "OK"}}}
        async with agent_client.stream(
            "POST",
            "/copilotkit",
            json=agui_request(thread_id, forwarded_props=resume_props),
            headers={"Accept": "text/event-stream"},
        ) as sse_resp:
            await sse_resp.aread()

        assert webhook_route.called, "Agent should POST to Unity webhook after finalisation"

        posted = json.loads(webhook_route.calls.last.request.content)
        assert posted["case_id"] == "CASE-E2E-WEBHOOK"
        assert posted["hitl_decision"] == "approve"
        assert posted["status"] == "complete"
        assert posted["client_email"] == "iris@vip.example.com"

    @pytest.mark.asyncio
    async def test_full_approve_flow(self, clients, respx_mock):
        """
        Smoke test for the complete happy path:
          intake → interrupt → reconnect sees interrupt → approve → webhook fired.
        """
        agent_client, unity_client, isolated_graph = clients

        webhook_route = respx_mock.post("http://unity/webhook/case-update").respond(
            200, json={"received": True}
        )

        # 1. Unity submits case
        resp = await agent_client.post(
            "/api/cases",
            json={
                "client_email": "jane@billing.example.com",
                "unity_webhook_url": "http://unity/webhook/case-update",
                "case_id": "CASE-FULL-01",
            },
        )
        assert resp.status_code == 200
        thread_id = resp.json()["thread_id"]

        # 2. Graph is waiting for HITL
        snapshot = await isolated_graph.aget_state(
            {"configurable": {"thread_id": thread_id}}
        )
        assert "request_hitl_approval" in snapshot.next

        # 3. HITL frontend connects → sees interrupt
        async with agent_client.stream(
            "POST",
            "/copilotkit",
            json=agui_request(thread_id),
            headers={"Accept": "text/event-stream"},
        ) as sse_resp:
            raw = await sse_resp.aread()

        events = parse_sse_events(raw.decode())
        assert any(
            isinstance(e.get("data"), dict)
            and e["data"].get("type") == "CUSTOM"
            and "interrupt" in str(e["data"].get("name", "")).lower()
            for e in events
        ), "Must see on_interrupt CUSTOM event on connect"

        # 4. User approves
        resume_props = {"command": {"resume": {"decision": "approve", "notes": "All good"}}}
        async with agent_client.stream(
            "POST",
            "/copilotkit",
            json=agui_request(thread_id, forwarded_props=resume_props),
            headers={"Accept": "text/event-stream"},
        ) as sse_resp:
            await sse_resp.aread()

        # 5. Webhook was called with correct payload
        assert webhook_route.called
        body = json.loads(webhook_route.calls.last.request.content)
        assert body["status"] == "complete"
        assert body["hitl_decision"] == "approve"

        # 6. Graph is done
        snapshot = await isolated_graph.aget_state(
            {"configurable": {"thread_id": thread_id}}
        )
        assert snapshot.next == ()

    @pytest.mark.asyncio
    async def test_full_reject_flow(self, clients, respx_mock):
        """Rejection path: agent still finalises and posts to webhook."""
        agent_client, _, isolated_graph = clients

        webhook_route = respx_mock.post("http://unity/webhook/case-update").respond(
            200, json={"received": True}
        )

        resp = await agent_client.post(
            "/api/cases",
            json={
                "client_email": "kate@fraud.example.com",
                "unity_webhook_url": "http://unity/webhook/case-update",
                "case_id": "CASE-FULL-REJECT",
            },
        )
        thread_id = resp.json()["thread_id"]

        # Reject via AG-UI
        resume_props = {"command": {"resume": {"decision": "reject", "notes": "Needs review"}}}
        async with agent_client.stream(
            "POST",
            "/copilotkit",
            json=agui_request(thread_id, forwarded_props=resume_props),
            headers={"Accept": "text/event-stream"},
        ) as sse_resp:
            await sse_resp.aread()

        assert webhook_route.called
        body = json.loads(webhook_route.calls.last.request.content)
        assert body["hitl_decision"] == "reject"
        assert body["status"] == "complete"

        snapshot = await isolated_graph.aget_state(
            {"configurable": {"thread_id": thread_id}}
        )
        assert snapshot.next == ()
