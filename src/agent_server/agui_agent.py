"""
StateSyncAGUIAgent – fixes the reconnect fast-path in ag_ui_langgraph.

Problem
-------
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

Fix
---
We override ``prepare_stream``.  After calling super() we detect whether the
fast-path was taken (``events_to_dispatch`` is not None) and, if so, inject
a ``STATE_SNAPSHOT`` (and ``MESSAGES_SNAPSHOT`` for message history) into the
event list *before* the ``on_interrupt`` event.

Result
------
On reconnect:

    RUN_STARTED → STATE_SNAPSHOT → MESSAGES_SNAPSHOT → on_interrupt → RUN_FINISHED

The frontend now receives the full agent state automatically.  The interrupt
payload only needs to carry what is *specific to the decision* (the question
and the recommended action), not a redundant copy of the entire state.

Frontend implications
---------------------
* ``useCoAgent`` is populated on reconnect – sidebar, status indicators etc.
  all work without any extra fetching.
* ``useLangGraphInterrupt`` ``event.value`` carries the slim decision context:
  ``{ question, recommended_action, case_id }``.
* Message history (if the agent produces any) is restored automatically.
"""

from __future__ import annotations

from ag_ui.core import (
    EventType,
    MessagesSnapshotEvent,
    StateSnapshotEvent,
)
from ag_ui_langgraph.utils import langchain_messages_to_agui
from ag_ui_langgraph import LangGraphAgent
from langchain_core.runnables import RunnableConfig

from copilotkit import LangGraphAGUIAgent


class StateSyncAGUIAgent(LangGraphAGUIAgent):
    """
    ``LangGraphAGUIAgent`` with the reconnect fast-path patched to also emit
    ``STATE_SNAPSHOT`` and ``MESSAGES_SNAPSHOT``.
    """

    async def prepare_stream(
        self,
        input,          # RunAgentInput
        agent_state,    # StateSnapshot
        config: RunnableConfig,
    ) -> dict:
        result = await super().prepare_stream(input, agent_state, config)

        events_to_dispatch = result.get("events_to_dispatch")
        if not events_to_dispatch:
            # Not the fast-path (normal streaming run or resume).
            # STATE_SNAPSHOT is already emitted by the base class at stream end.
            return result

        # Fast-path: interrupted thread reconnect.
        # Build snapshot events from the checkpointed state.
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
