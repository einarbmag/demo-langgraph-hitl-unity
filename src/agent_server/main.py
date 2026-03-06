"""Entry point – run with:  uv run uvicorn agent_server.main:app --reload --port 8000"""
from agent_server.app import app  # noqa: F401  (re-exported for uvicorn)
