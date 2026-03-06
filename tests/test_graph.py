"""
Layer 1 – Direct graph tests (no HTTP, no AG-UI).

Tests verify the deterministic LangGraph flow:
  process_case → interrupt → resume → finalize_case → END
"""

from __future__ import annotations

import pytest
from langgraph.types import Command

from agent_server.graph import _derive_action


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def make_initial_state(
    *,
    client_email: str = "alice@billing.example.com",
    unity_webhook_url: str = "http://unity/webhook/case-update",
    case_id: str = "CASE-001",
) -> dict:
    return {
        "case_id": case_id,
        "client_email": client_email,
        "unity_webhook_url": unity_webhook_url,
        "messages": [],
    }


def thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


# ---------------------------------------------------------------------------
# _derive_action – determinism
# ---------------------------------------------------------------------------

class TestDeriveAction:
    def test_billing_keyword(self):
        assert _derive_action("user@billing.co") == "issue_billing_credit"

    def test_vip_keyword(self):
        assert _derive_action("ceo@vip-corp.com") == "escalate_to_senior_agent"

    def test_fraud_keyword(self):
        assert _derive_action("report@fraud-alerts.net") == "flag_for_fraud_review"

    def test_fallback_is_stable(self):
        email = "random@example.com"
        assert _derive_action(email) == _derive_action(email)

    def test_fallback_is_one_of_known_actions(self):
        from agent_server.graph import _ACTION_MAP
        action = _derive_action("unknown@example.com")
        assert action in _ACTION_MAP.values()


# ---------------------------------------------------------------------------
# Graph: process_case
# ---------------------------------------------------------------------------

class TestProcessCase:
    @pytest.mark.asyncio
    async def test_process_case_populates_summary(self, case_graph):
        state = make_initial_state(client_email="alice@billing.example.com")
        result = await case_graph.ainvoke(state, config=thread_config("t-pc-1"))

        # result has __interrupt__ because the graph pauses at request_hitl_approval
        assert "__interrupt__" in result
        # But the state stored in the checkpoint should have the summary
        snapshot = await case_graph.aget_state(thread_config("t-pc-1"))
        assert "alice@billing.example.com" in snapshot.values["case_summary"]
        assert snapshot.values["recommended_action"] == "issue_billing_credit"

    @pytest.mark.asyncio
    async def test_process_case_sets_status_awaiting_approval(self, case_graph):
        state = make_initial_state()
        await case_graph.ainvoke(state, config=thread_config("t-pc-2"))
        snapshot = await case_graph.aget_state(thread_config("t-pc-2"))
        assert snapshot.values["status"] == "awaiting_approval"


# ---------------------------------------------------------------------------
# Graph: interrupt at request_hitl_approval
# ---------------------------------------------------------------------------

class TestHITLInterrupt:
    @pytest.mark.asyncio
    async def test_graph_pauses_at_interrupt(self, case_graph):
        state = make_initial_state()
        result = await case_graph.ainvoke(state, config=thread_config("t-int-1"))

        assert "__interrupt__" in result
        interrupts = result["__interrupt__"]
        assert len(interrupts) == 1

    @pytest.mark.asyncio
    async def test_interrupt_payload_contains_case_fields(self, case_graph):
        state = make_initial_state(
            client_email="bob@fraud.example.com",
            case_id="CASE-042",
        )
        result = await case_graph.ainvoke(state, config=thread_config("t-int-2"))
        payload = result["__interrupt__"][0].value

        # Decision-specific fields are present in the interrupt payload
        assert payload["case_id"] == "CASE-042"
        assert payload["recommended_action"] == "flag_for_fraud_review"
        assert "question" in payload
        # Full state fields (case_summary, client_email) are NOT in the payload –
        # they arrive via STATE_SNAPSHOT so useCoAgent is populated automatically.
        assert "client_email" not in payload
        assert "case_summary" not in payload

    @pytest.mark.asyncio
    async def test_graph_next_node_is_approval_after_interrupt(self, case_graph):
        state = make_initial_state()
        await case_graph.ainvoke(state, config=thread_config("t-int-3"))
        snapshot = await case_graph.aget_state(thread_config("t-int-3"))

        # The graph is paused inside request_hitl_approval
        assert "request_hitl_approval" in snapshot.next

    @pytest.mark.asyncio
    async def test_reconnecting_to_thread_exposes_interrupt(self, case_graph):
        """
        After Unity submits a case the graph pauses.  When the HITL frontend
        later reconnects using the same thread_id it should find the interrupt
        still pending in the checkpoint.
        """
        cfg = thread_config("t-int-4")
        await case_graph.ainvoke(make_initial_state(), config=cfg)

        # Simulates reconnect: get_state on the same thread_id
        snapshot = await case_graph.aget_state(cfg)
        interrupts = snapshot.tasks[0].interrupts if snapshot.tasks else []

        assert len(interrupts) == 1
        assert "recommended_action" in interrupts[0].value


# ---------------------------------------------------------------------------
# Graph: resume with approval
# ---------------------------------------------------------------------------

class TestResumeApproval:
    @pytest.mark.asyncio
    async def test_approve_sets_hitl_decision(self, case_graph, respx_mock):
        cfg = thread_config("t-res-approve-1")
        await case_graph.ainvoke(make_initial_state(), config=cfg)

        respx_mock.post("http://unity/webhook/case-update").respond(200, json={"received": True})

        result = await case_graph.ainvoke(
            Command(resume={"decision": "approve", "notes": "Looks good"}),
            config=cfg,
        )

        snapshot = await case_graph.aget_state(cfg)
        assert snapshot.values["hitl_decision"] == "approve"
        assert snapshot.values["hitl_notes"] == "Looks good"

    @pytest.mark.asyncio
    async def test_approve_reaches_end(self, case_graph, respx_mock):
        cfg = thread_config("t-res-approve-2")
        await case_graph.ainvoke(make_initial_state(), config=cfg)

        respx_mock.post("http://unity/webhook/case-update").respond(200, json={"received": True})

        await case_graph.ainvoke(
            Command(resume={"decision": "approve", "notes": ""}),
            config=cfg,
        )

        snapshot = await case_graph.aget_state(cfg)
        # Graph is at END – next is empty
        assert snapshot.next == ()
        assert snapshot.values["status"] == "complete"

    @pytest.mark.asyncio
    async def test_approve_posts_to_unity_webhook(self, case_graph, respx_mock):
        cfg = thread_config("t-res-approve-3")
        await case_graph.ainvoke(
            make_initial_state(unity_webhook_url="http://unity/webhook/case-update"),
            config=cfg,
        )

        webhook_route = respx_mock.post("http://unity/webhook/case-update").respond(
            200, json={"received": True}
        )

        await case_graph.ainvoke(
            Command(resume={"decision": "approve", "notes": "All clear"}),
            config=cfg,
        )

        assert webhook_route.called
        posted_json = webhook_route.calls.last.request.content
        import json
        body = json.loads(posted_json)
        assert body["hitl_decision"] == "approve"
        assert body["status"] == "complete"


# ---------------------------------------------------------------------------
# Graph: resume with rejection
# ---------------------------------------------------------------------------

class TestResumeRejection:
    @pytest.mark.asyncio
    async def test_reject_sets_hitl_decision(self, case_graph, respx_mock):
        cfg = thread_config("t-res-reject-1")
        await case_graph.ainvoke(make_initial_state(), config=cfg)

        respx_mock.post("http://unity/webhook/case-update").respond(200, json={"received": True})

        await case_graph.ainvoke(
            Command(resume={"decision": "reject", "notes": "Needs more info"}),
            config=cfg,
        )

        snapshot = await case_graph.aget_state(cfg)
        assert snapshot.values["hitl_decision"] == "reject"
        assert snapshot.values["hitl_notes"] == "Needs more info"

    @pytest.mark.asyncio
    async def test_reject_still_posts_to_unity(self, case_graph, respx_mock):
        cfg = thread_config("t-res-reject-2")
        await case_graph.ainvoke(make_initial_state(), config=cfg)

        webhook_route = respx_mock.post("http://unity/webhook/case-update").respond(
            200, json={"received": True}
        )

        await case_graph.ainvoke(
            Command(resume={"decision": "reject", "notes": "On hold"}),
            config=cfg,
        )

        assert webhook_route.called
        import json
        body = json.loads(webhook_route.calls.last.request.content)
        assert body["hitl_decision"] == "reject"
        assert body["status"] == "complete"
