# LangGraph + CopilotKit HITL — Unity Case Management Demo

A working demonstration of a LangGraph agent that:

1. Accepts cases from an external case management system (Unity) over HTTP
2. Does deterministic processing, then **pauses for human-in-the-loop (HITL) approval**
3. Serves a CopilotKit AG-UI endpoint so a React frontend can connect to any paused thread by `thread_id`, immediately see the agent state and pending approval request, and resolve it
4. Calls back to Unity with the outcome after the human decides

All behaviour is verified by 34 automated tests — no LLM calls, no real servers required.

---

## Sequence

```
Unity                Agent Server              HITL Frontend (React)
  │                       │                           │
  │  POST /api/cases       │                           │
  │  { client_email, … }  │                           │
  │──────────────────────>│                           │
  │                       │  process_case node        │
  │                       │  ──────────────────       │
  │                       │  interrupt() checkpoint   │
  │  { thread_id }        │                           │
  │<──────────────────────│                           │
  │                       │                           │
  │  (opens case in UI,   │                           │
  │   passes thread_id    │                           │
  │   as URL param)       │                           │
  │                       │                           │
  │                       │  POST /copilotkit         │
  │                       │  { thread_id }            │
  │                       │<──────────────────────────│
  │                       │  SSE stream:              │
  │                       │   STATE_SNAPSHOT ──────>  │  useCoAgent populated
  │                       │   MESSAGES_SNAPSHOT ───>  │  message history restored
  │                       │   on_interrupt ────────>  │  useLangGraphInterrupt fires
  │                       │   RUN_FINISHED ─────────> │
  │                       │                           │
  │                       │  (user approves/rejects)  │
  │                       │                           │
  │                       │  POST /copilotkit         │
  │                       │  { forwarded_props:       │
  │                       │    command.resume: {…} }  │
  │                       │<──────────────────────────│
  │                       │  finalize_case node       │
  │                       │  ──────────────────       │
  │  POST /webhook/…      │                           │
  │<──────────────────────│                           │
  │  { decision, status } │                           │
```

---

## Project structure

```
src/agent_server/
  graph.py          LangGraph graph: CaseState + 3 nodes + shared MemorySaver
  agui_agent.py     StateSyncAGUIAgent: fixes the AG-UI reconnect fast-path
  app.py            FastAPI: POST /api/cases (Unity intake) + POST /copilotkit (AG-UI)
  main.py           Uvicorn entry point

mock_unity/
  app.py            Mock Unity webhook receiver (POST/GET /webhook/case-update)

tests/
  conftest.py       Shared fixtures (isolated graphs, ASGI clients)
  test_graph.py     16 tests – graph logic, interrupt, resume, webhook callback
  test_intake.py     5 tests – Unity intake HTTP endpoint
  test_agui_sync.py  5 tests – AG-UI SSE stream and state-sync behaviour
  test_full_flow.py  8 tests – complete end-to-end sequence
```

---

## Running the tests

```bash
uv sync --extra dev
uv run pytest -v
# 34 passed
```

## Running the server

```bash
uv run uvicorn agent_server.main:app --reload --port 8000
```

Endpoints:

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/api/cases` | Unity submits a case; returns `thread_id` |
| `POST` | `/copilotkit` | CopilotKit AG-UI SSE stream |
| `GET`  | `/copilotkit/health` | Liveness check |

---

## Key findings

### 1. How LangGraph HITL and thread persistence work

`interrupt(payload)` inside a node pauses graph execution and saves a checkpoint under the current `thread_id`. The graph resumes when invoked again with `Command(resume=value)` using the same `thread_id`. The checkpoint persists as long as the `MemorySaver` (or `PostgresSaver` in production) is in scope.

**Design choice for the Unity intake endpoint:** `graph.ainvoke()` is called synchronously. It runs `process_case`, reaches `interrupt()` in `request_hitl_approval`, saves the checkpoint, and returns. The entire processing up to the interrupt happens in the HTTP request handler — it's fast enough for a deterministic graph. The returned `thread_id` is what Unity stores against the case.

### 2. How AG-UI state sync works on reconnect

When the HITL frontend loads (with a `thread_id` from the URL), it calls `POST /copilotkit` with that `thread_id`. The `ag_ui_langgraph` library detects the active interrupt in the checkpoint and takes a **reconnect fast-path**:

```
RUN_STARTED → on_interrupt (CUSTOM) → RUN_FINISHED
```

This fires `useLangGraphInterrupt` immediately — the user sees the approval UI without any graph re-execution. **No polling required.**

### 3. The fast-path gap in `ag_ui_langgraph` (and the fix)

The reconnect fast-path skips `STATE_SNAPSHOT` and `MESSAGES_SNAPSHOT`. The consequences without a fix:

- `useCoAgent` state is **never populated** — you would have to manually duplicate all agent state fields inside the `interrupt()` payload just to render the UI
- Message history is **not restored**

**`StateSyncAGUIAgent`** (`src/agent_server/agui_agent.py`) fixes this by subclassing `LangGraphAGUIAgent` and overriding `prepare_stream` to inject both events into the fast-path:

```
RUN_STARTED → STATE_SNAPSHOT → MESSAGES_SNAPSHOT → on_interrupt → RUN_FINISHED
```

The `STATE_SNAPSHOT` contains the full checkpoint state (`case_id`, `case_summary`, `recommended_action`, `client_email`, `status`, …). The `MESSAGES_SNAPSHOT` carries the LangChain message history. Both are populated automatically on the frontend via `useCoAgent` and `CopilotChat` — no manual state duplication needed.

The fix is ~25 lines, purely additive, and requires no changes to the graph or the frontend:

```python
class StateSyncAGUIAgent(LangGraphAGUIAgent):
    async def prepare_stream(self, input, agent_state, config):
        result = await super().prepare_stream(input, agent_state, config)
        if not result.get("events_to_dispatch"):   # only act on the fast-path
            return result
        state_values = agent_state.values
        events_to_dispatch = result["events_to_dispatch"]
        events_to_dispatch.insert(1, StateSnapshotEvent(...))
        events_to_dispatch.insert(1, MessagesSnapshotEvent(...))
        return result
```

### 4. Interrupt payload design

Because `STATE_SNAPSHOT` is now emitted automatically, the `interrupt()` payload only needs to carry what is **specific to the decision**:

```python
# Before the fix — had to duplicate state manually
interrupt({
    "question": "Approve or reject?",
    "case_id": state["case_id"],
    "client_email": state["client_email"],   # ← redundant
    "case_summary": state["case_summary"],   # ← redundant
    "recommended_action": state["recommended_action"],
})

# After the fix — only the decision context
interrupt({
    "question": "Approve or reject?",
    "case_id": state["case_id"],
    "recommended_action": state["recommended_action"],
})
```

### 5. Resume via AG-UI

The HITL frontend resolves the interrupt by posting back to the same `/copilotkit` endpoint with `forwarded_props.command.resume`:

```json
{
  "thread_id": "<same thread_id>",
  "forwarded_props": {
    "command": {
      "resume": { "decision": "approve", "notes": "Looks good" }
    }
  }
}
```

`ag_ui_langgraph` detects the `resume` key, wraps it in `Command(resume=...)`, and invokes the graph. The value becomes the return value of `interrupt()` inside `request_hitl_approval`, the node completes, `finalize_case` runs, and the Unity webhook is called.

---

## Frontend wiring (React / Next.js)

The `thread_id` comes from the URL parameter that Unity embeds in the iframe:

```tsx
// pages/hitl.tsx
const threadId = new URLSearchParams(window.location.search).get("thread")

<CopilotKit runtimeUrl="/copilotkit" threadId={threadId}>
  <CaseReviewPanel />
</CopilotKit>
```

Inside the panel:

```tsx
function CaseReviewPanel() {
  // Populated automatically via STATE_SNAPSHOT on connect
  const { state } = useCoAgent<CaseState>({ name: "case_agent" })

  // Fires immediately if the thread has a pending interrupt
  useLangGraphInterrupt<{ question: string; recommended_action: string }>({
    render: ({ event, resolve }) => (
      <div>
        {/* Full state available from useCoAgent — no manual copying needed */}
        <h2>Case {state.case_id}</h2>
        <p>{state.case_summary}</p>
        <p>Client: {state.client_email}</p>

        {/* Decision context from the interrupt payload */}
        <p>{event.value.question}</p>
        <p>Recommended action: <strong>{event.value.recommended_action}</strong></p>

        <button onClick={() => resolve({ decision: "approve", notes: "" })}>
          Approve
        </button>
        <button onClick={() => resolve({ decision: "reject", notes: "" })}>
          Reject
        </button>
      </div>
    ),
  })

  return <CopilotChat />   // message history restored via MESSAGES_SNAPSHOT
}
```

---

## Production notes

| Topic | Recommendation |
|-------|----------------|
| **Checkpointer** | Replace `MemorySaver` with `AsyncPostgresSaver` so thread state survives server restarts |
| **Thread ID storage** | Unity should persist the `thread_id` against the case record in its own database |
| **Webhook security** | Add an `Authorization` header or HMAC signature to the Unity webhook POST |
| **Agent server auth** | Put the `/api/cases` intake endpoint behind an API key so only Unity can submit cases |
| **Multiple agents** | `StateSyncAGUIAgent` is generic — it works for any LangGraph graph with `interrupt()` |
