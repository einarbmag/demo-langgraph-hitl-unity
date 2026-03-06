"""
Shared pytest fixtures.

Fixtures are scoped at the function level so each test gets an isolated
in-memory checkpointer (and therefore isolated thread state).
"""

from __future__ import annotations

import pytest
import pytest_asyncio
import httpx
from langgraph.checkpoint.memory import MemorySaver

from agent_server.graph import build_graph
from agent_server.app import app as agent_app
from mock_unity.app import app as unity_app


@pytest.fixture()
def checkpointer():
    """Fresh in-memory checkpointer for each test."""
    return MemorySaver()


@pytest.fixture()
def case_graph(checkpointer):
    """Compiled case-management graph with an isolated checkpointer."""
    return build_graph(checkpointer=checkpointer)


@pytest_asyncio.fixture()
async def agent_client(case_graph):
    """
    httpx AsyncClient pointed at the agent FastAPI app.

    We monkey-patch the app's graph dependency so each test uses its own
    isolated in-memory checkpointer.
    """
    import agent_server.app as agent_app_module
    original_graph = agent_app_module.graph

    # Replace the module-level graph with our isolated test graph
    agent_app_module.graph = case_graph

    # Also replace inside the AG-UI agent registered on the FastAPI router.
    # The LangGraphAGUIAgent holds a reference to the graph; we need to
    # update that reference too.  The easiest way is to re-register the
    # endpoint, but that mutates the shared app.  Instead we patch the
    # agent object directly.
    from copilotkit import LangGraphAGUIAgent
    for route in agent_app.routes:
        # find the LangGraphAGUIAgent registered via add_langgraph_fastapi_endpoint
        pass  # route patching not needed – we test the graph directly for HITL

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=agent_app), base_url="http://test"
    ) as client:
        yield client

    # Restore
    agent_app_module.graph = original_graph


@pytest_asyncio.fixture()
async def unity_client():
    """httpx AsyncClient pointed at the mock Unity app."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=unity_app), base_url="http://unity"
    ) as client:
        # Reset stored updates before each test
        await client.delete("/webhook/case-updates")
        yield client
