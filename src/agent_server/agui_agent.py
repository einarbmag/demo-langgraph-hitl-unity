"""
StateSyncAGUIAgent – fixes two gaps in ag_ui_langgraph.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Gap 1 – Interrupt reconnect fast-path missing state sync
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
When the HITL frontend connects to a thread that is paused at an interrupt(),
``ag_ui_langgraph``'s ``prepare_stream`` takes a fast-path that only emits::

    RUN_STARTED → on_interrupt (CUSTOM) → RUN_FINISHED

It skips ``STATE_SNAPSHOT`` and ``MESSAGES_SNAPSHOT``.  The consequence is
that:

* ``useCoAgent`` state is *never populated* on reconnect – the frontend must
  manually duplicate all relevant state inside the interrupt() payload just to
  render the approval UI.
* Message history is not restored, so ``CopilotChat`` would show an empty
  thread.

Fix: override ``prepare_stream``.  After calling super() we detect whether the
fast-path was taken (``events_to_dispatch`` is not None) and, if so, inject
a ``STATE_SNAPSHOT`` (and ``MESSAGES_SNAPSHOT`` for message history) into the
event list *before* the ``on_interrupt`` event.

Result on reconnect::

    RUN_STARTED → STATE_SNAPSHOT → MESSAGES_SNAPSHOT → on_interrupt → RUN_FINISHED

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Gap 2 – Completed-thread triggers unintended graph re-run
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
When the frontend connects to a thread whose graph has already reached END
(status="complete", no pending interrupt), the base library's default path
calls ``graph.astream_events(input=state)`` – which LangGraph interprets as
a *new run from START*.  This re-executes process_case → request_hitl_approval
→ interrupt, overwriting the completed case with a fresh one.

Fix: detect the completed-thread case (``next == ()`` with non-empty state and
no resume command) and immediately emit the final state without touching the
graph::

    RUN_STARTED → STATE_SNAPSHOT → MESSAGES_SNAPSHOT → RUN_FINISHED

This lets auditors or supervisors open the HITL UI after a case has been
resolved and see the final decision and state history without triggering any
new execution.

Frontend implications
---------------------
* ``useCoAgent`` is populated on reconnect for both paused and completed threads.
* ``useLangGraphInterrupt`` only fires for paused threads (there is no
  ``on_interrupt`` event for completed threads).
* ``CopilotChat`` shows message history in both cases.
"""

from __future__ import annotations

from ag_ui.core import (
    EventType,
    MessagesSnapshotEvent,
    RunFinishedEvent,
    StateSnapshotEvent,
)
from ag_ui_langgraph.utils import langchain_messages_to_agui
from ag_ui_langgraph import LangGraphAgent
from langchain_core.runnables import RunnableConfig

from copilotkit import LangGraphAGUIAgent


class StateSyncAGUIAgent(LangGraphAGUIAgent):
    """
    ``LangGraphAGUIAgent`` with two reconnect gaps patched:

    1. Interrupted thread: injects ``STATE_SNAPSHOT`` + ``MESSAGES_SNAPSHOT``
       before the ``on_interrupt`` event so ``useCoAgent`` is populated.

    2. Completed thread: emits the final state directly instead of re-running
       the graph from scratch.
    """

    async def prepare_stream(
        self,
        input,          # RunAgentInput
        agent_state,    # StateSnapshot
        config: RunnableConfig,
    ) -> dict:
        result = await super().prepare_stream(input, agent_state, config)

        events_to_dispatch = result.get("events_to_dispatch")

        # ── Gap 1: Interrupt reconnect fast-path ─────────────────────────────
        if events_to_dispatch:
            # ag_ui_langgraph detected a pending interrupt and will replay it.
            # Inject STATE_SNAPSHOT + MESSAGES_SNAPSHOT so that useCoAgent
            # is populated and CopilotChat shows message history.
            state_values = agent_state.values

            state_snapshot = StateSnapshotEvent(
                type=EventType.STATE_SNAPSHOT,
                snapshot=self.get_state_snapshot(state_values),
            )
            messages_snapshot = MessagesSnapshotEvent(
                type=EventType.MESSAGES_SNAPSHOT,
                messages=langchain_messages_to_agui(state_values.get("messages", [])),
            )

            # Insert after RUN_STARTED (index 0), before on_interrupt.
            # Final order: RunStarted, STATE_SNAPSHOT, MESSAGES_SNAPSHOT, on_interrupt, RunFinished
            events_to_dispatch.insert(1, messages_snapshot)
            events_to_dispatch.insert(1, state_snapshot)

            return result

        # ── Gap 2: Completed-thread re-run prevention ─────────────────────────
        #
        # If the graph reached END (next is empty, state is non-empty) and the
        # frontend is not sending a resume command, the base library would
        # re-invoke graph.astream_events() with the current state as input.
        # LangGraph interprets this as a *new run from START*, which re-runs
        # process_case → request_hitl_approval → new interrupt — overwriting
        # the completed case.
        #
        # We short-circuit this by emitting the final checkpointed state
        # directly via the fast-path mechanism (events_to_dispatch), so
        # _handle_stream_events skips the streaming invocation entirely.
        forwarded_props = input.forwarded_props or {}
        resume_input = forwarded_props.get("command", {}).get("resume", None)
        is_completed_thread = (
            not resume_input
            and not agent_state.next          # () → graph at END
            and bool(agent_state.values)      # non-empty → has been run before
        )

        if is_completed_thread:
            thread_id = input.thread_id
            run_id = self.active_run["id"]
            state_values = agent_state.values

            # _handle_stream_events always emits RUN_STARTED before dispatching
            # events_to_dispatch, so we don't include it here to avoid doubling.
            # We DO need RUN_FINISHED because the early return in
            # _handle_stream_events skips the normal end-of-stream emission.
            return {
                "stream": None,
                "state": None,
                "config": None,
                "events_to_dispatch": [
                    StateSnapshotEvent(
                        type=EventType.STATE_SNAPSHOT,
                        snapshot=self.get_state_snapshot(state_values),
                    ),
                    MessagesSnapshotEvent(
                        type=EventType.MESSAGES_SNAPSHOT,
                        messages=langchain_messages_to_agui(state_values.get("messages", [])),
                    ),
                    RunFinishedEvent(
                        type=EventType.RUN_FINISHED,
                        thread_id=thread_id,
                        run_id=run_id,
                    ),
                ],
            }

        return result
