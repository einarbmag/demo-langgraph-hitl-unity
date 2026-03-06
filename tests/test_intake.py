"""
Layer 2 – Unity intake endpoint tests (HTTP).

Tests verify that Unity can submit a case via POST /api/cases and receive a
thread_id back, and that the agent is left in an interrupted state ready for
the HITL frontend to connect.
"""

from __future__ import annotations

import pytest
import httpx

from agent_server.app import app as agent_app


@pytest.fixture()
async def fresh_agent_client(checkpointer):
    """
    Agent client wired to an isolated graph so each test is independent.
    We need to patch both the module-level graph (used by the intake
    endpoint) and the LangGraphAGUIAgent inside the router.
    """
    import agent_server.app as mod
    from agent_server.graph import build_graph

    isolated_graph = build_graph(checkpointer=checkpointer)
    original = mod.graph
    mod.graph = isolated_graph

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=agent_app), base_url="http://test"
    ) as client:
        yield client, isolated_graph

    mod.graph = original


class TestIntakeEndpoint:
    @pytest.mark.asyncio
    async def test_returns_thread_id_and_case_id(self, fresh_agent_client):
        client, _ = fresh_agent_client
        resp = await client.post(
            "/api/cases",
            json={
                "client_email": "alice@billing.example.com",
                "unity_webhook_url": "http://unity/webhook/case-update",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "thread_id" in body
        assert "case_id" in body
        assert len(body["thread_id"]) == 36   # UUID format
        assert len(body["case_id"]) == 36

    @pytest.mark.asyncio
    async def test_accepts_explicit_case_id(self, fresh_agent_client):
        client, _ = fresh_agent_client
        resp = await client.post(
            "/api/cases",
            json={
                "client_email": "bob@vip.example.com",
                "unity_webhook_url": "http://unity/webhook/case-update",
                "case_id": "CASE-EXPLICIT-99",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["case_id"] == "CASE-EXPLICIT-99"

    @pytest.mark.asyncio
    async def test_graph_is_in_interrupted_state_after_intake(self, fresh_agent_client, checkpointer):
        """After intake the graph checkpoint should show a pending interrupt."""
        client, isolated_graph = fresh_agent_client
        resp = await client.post(
            "/api/cases",
            json={
                "client_email": "carol@fraud.example.com",
                "unity_webhook_url": "http://unity/webhook/case-update",
            },
        )
        thread_id = resp.json()["thread_id"]

        snapshot = await isolated_graph.aget_state(
            {"configurable": {"thread_id": thread_id}}
        )
        # The graph is paused inside request_hitl_approval
        assert "request_hitl_approval" in snapshot.next
        interrupts = snapshot.tasks[0].interrupts if snapshot.tasks else []
        assert len(interrupts) == 1

    @pytest.mark.asyncio
    async def test_interrupt_payload_matches_case(self, fresh_agent_client):
        client, isolated_graph = fresh_agent_client
        resp = await client.post(
            "/api/cases",
            json={
                "client_email": "dave@vip.example.com",
                "unity_webhook_url": "http://unity/webhook/case-update",
                "case_id": "CASE-VIP-01",
            },
        )
        thread_id = resp.json()["thread_id"]

        snapshot = await isolated_graph.aget_state(
            {"configurable": {"thread_id": thread_id}}
        )
        interrupt_value = snapshot.tasks[0].interrupts[0].value
        # Decision-specific fields present in the interrupt payload
        assert interrupt_value["case_id"] == "CASE-VIP-01"
        assert interrupt_value["recommended_action"] == "escalate_to_senior_agent"
        # Full state fields are NOT duplicated; they arrive via STATE_SNAPSHOT
        assert "client_email" not in interrupt_value
        assert "case_summary" not in interrupt_value

    @pytest.mark.asyncio
    async def test_two_cases_get_different_thread_ids(self, fresh_agent_client):
        client, _ = fresh_agent_client
        payload = {
            "client_email": "x@billing.example.com",
            "unity_webhook_url": "http://unity/webhook/case-update",
        }
        r1 = await client.post("/api/cases", json=payload)
        r2 = await client.post("/api/cases", json=payload)

        assert r1.json()["thread_id"] != r2.json()["thread_id"]
