# Lab Report — LangGraph Agentic Orchestration

**Student:** Tran Van Dong (2A202601310)
**Date:** 2026-08-25 19:43
**Provider:** OpenAI (gpt-4o-mini)

---

## 1. Metrics Summary

| Metric | Value |
|--------|-------|
| Total Scenarios     | 7 |
| Success Rate        | 100.00% |
| Avg Nodes Visited   | 6.4 |
| Total Retries       | 3 |
| Total Interrupts    | 2 |
| Resume Success      | False |

---

## 2. Scenario Results

| Scenario ID | Expected | Actual | Success | Nodes | Retries | Approval | Errors |
|-------------|----------|--------|---------|-------|---------|----------|--------|
| S01_simple | simple | simple | ✅ | 4 | 0 | - | 0 |
| S02_tool | tool | tool | ✅ | 6 | 0 | - | 0 |
| S03_missing | missing_info | missing_info | ✅ | 4 | 0 | - | 0 |
| S04_risky | risky | risky | ✅ | 8 | 0 | req+obs | 0 |
| S05_error | error | error | ✅ | 10 | 2 | - | 2 |
| S06_delete | risky | risky | ✅ | 8 | 0 | req+obs | 0 |
| S07_dead_letter | error | error | ✅ | 5 | 1 | - | 1 |

---

## 3. Architecture

### Graph Design

The agent implements a **support-ticket routing graph** with 11 nodes and 4 conditional edges:

```
START → intake → classify → [route_after_classify]
  simple        → answer → finalize → END
  tool          → tool → evaluate → [route_after_evaluate]
                              success      → answer → finalize → END
                              needs_retry  → retry  → [route_after_retry]
                                                tool (attempt < max) → ...
                                                dead_letter          → finalize → END
  missing_info  → clarify → finalize → END
  risky         → risky_action → approval → [route_after_approval]
                                    approved → tool → evaluate → ...
                                    rejected → clarify → finalize → END
  error         → retry → [route_after_retry] → ...
```

### Node Register (11 nodes)

| Graph Name    | Function                  | Role                                |
|---------------|---------------------------|-------------------------------------|
| intake        | intake_node               | Normalize query                     |
| classify      | classify_node             | LLM structured-output classification|
| tool          | tool_node                 | Mock tool with error simulation     |
| evaluate      | evaluate_node             | Heuristic tool result check         |
| answer        | answer_node               | LLM grounded answer generation      |
| clarify       | ask_clarification_node    | Clarification question              |
| risky_action  | risky_action_node         | Propose action (no side effect)     |
| approval      | approval_node             | Mock HITL gate                      |
| retry         | retry_or_fallback_node    | Increment attempt counter           |
| dead_letter   | dead_letter_node          | Escalation after max retries        |
| finalize      | finalize_node             | Terminal audit event                |

### State Schema

| Field              | Type             | Update Strategy |
|--------------------|------------------|-----------------|
| thread_id          | str              | Overwrite (init)|
| scenario_id        | str              | Overwrite (init)|
| query              | str              | Overwrite       |
| route              | str              | Overwrite       |
| risk_level         | str              | Overwrite       |
| attempt            | int              | Overwrite       |
| max_attempts       | int              | Overwrite (init)|
| final_answer       | str \| None      | Overwrite       |
| evaluation_result  | str \| None      | Overwrite       |
| pending_question   | str \| None      | Overwrite       |
| proposed_action    | str \| None      | Overwrite       |
| approval           | dict \| None     | Overwrite       |
| messages           | list[str]        | **Append** (add)|
| tool_results       | list[str]        | **Append** (add)|
| errors             | list[str]        | **Append** (add)|
| events             | list[dict]       | **Append** (add)|

**Rationale:** Scalar fields represent the current state of one execution dimension and should be
overwritten (e.g., `attempt` is the count, not a history). The four list fields are append-only
because they form the audit trail and must never lose history. The `add` reducer ensures LangGraph
merges partial updates correctly without node mutation.

### LLM Integration

- **classify_node**: Uses `llm.with_structured_output(ClassifyOutput)` to enforce a Pydantic schema
  with `route`, `risk_level`, and `reasoning`. Priority prompt: risky > tool > missing_info > error > simple.
- **answer_node**: Uses `llm.invoke()` with system prompt + context (query + tool results + approval).
  Falls back to a safe escalation message if the LLM call fails, logging the error.

---

## 4. Failure Analysis

### Failure Mode 1: Tool Failure → Bounded Retry → Dead Letter

**Scenario:** S05_error / S07_dead_letter  
**Signal:** `tool_results` latest entry contains "ERROR".  
**Detection:** `evaluate_node` checks for "ERROR" substring → `evaluation_result = "needs_retry"`.  
**Route:** `route_after_evaluate` → "retry" → `retry_or_fallback_node` increments `attempt`.  
`route_after_retry` routes to "tool" if `attempt < max_attempts`, else "dead_letter".  
**S07 trace:** `max_attempts=1`. After first retry, `attempt=1 >= 1` → `dead_letter` → `finalize`.  
**Termination guarantee:** The bounded check `attempt >= max_attempts` prevents infinite loops.  
**Residual risk:** The heuristic "ERROR substring" check may miss non-standard error formats;
an LLM-as-judge evaluator would be more robust.

### Failure Mode 2: Risky Action Rejected by Approver

**Scenario:** S04_risky / S06_delete (rejected branch)  
**Signal:** `approval.approved == False`.  
**Detection:** `route_after_approval` reads `approval.get("approved")`.  
**Route:** "clarify" → `ask_clarification_node` reads `approval.comment` + `proposed_action`
and generates an actionable question explaining the rejection.  
**Termination guarantee:** `clarify` always leads to `finalize → END` via fixed edge.  
**Containment:** The tool node is never reached; the risky side effect is never executed.  
**Residual risk:** The mock approver always approves in CI; real rejections require LANGGRAPH_INTERRUPT=true.

---

## 5. Persistence and Recovery Evidence

### MemorySaver (Default)
- **Checkpointer:** `MemorySaver` is attached to the compiled graph via `build_graph(checkpointer=...)`.
- **Thread ID:** Each scenario gets a unique `thread_id = "thread-{scenario_id}"` from `initial_state()`.
  The CLI passes it as `{"configurable": {"thread_id": state["thread_id"]}}`.
- **State History:** Within a process, `graph.get_state_history(config)` shows all checkpoints:
  ```
  Thread ID: thread-persist-demo
  Checkpoints recorded: 6
    Checkpoint 0: 4 events, next=()
    Checkpoint 1: 3 events, next=('finalize',)
    Checkpoint 2: 2 events, next=('answer',)
  ```
- **Limitation:** MemorySaver is in-process only. State is lost when the process exits.

### SQLite (Extension — Crash-Resume Evidence)
- **Backend:** `SqliteSaver` with WAL mode enabled via `build_checkpointer("sqlite", path)`.
- **Crash-Resume Proof:** After running a graph with SQLite, a new `SqliteSaver` connection was opened
  to the same `.db` file (simulating a process restart). The state history survived:
  ```
  Thread ID:   thread-sqlite-demo
  DB file:     checkpoints_evidence.db
  Checkpoints after simulated restart: 8
  [OK] SQLite: state history survived process-level reconnect (crash-resume evidence).
  ```
- **Conclusion:** SQLite persistence is durable across process-level reconnects. A full crash-resume
  would require `graph.invoke(None, config=config)` to resume from the last checkpoint.

---

## 6. Extension Work

- **SQLite persistence**: `build_checkpointer("sqlite", path)` fully implemented with WAL mode.
  Crash-resume evidence collected (8 checkpoints survived reconnect).
- **Real latency instrumentation**: `cli.py` measures wall-clock time per scenario via
  `time.perf_counter()` and patches it into `ScenarioMetric.latency_ms`.
- **LLM-as-judge** for evaluate_node: not implemented (heuristic used for base score).
- **Real HITL interrupt**: LANGGRAPH_INTERRUPT=true env flag triggers `interrupt()` in approval_node;
  requires manual resume via `graph.invoke(None, config=...)`.

---

## 7. Improvement Plan

**Priority 1: Replace heuristic evaluator with LLM-as-judge**
The current ERROR-substring check is brittle. A structured LLM call with a `verdict` + `reason`
schema would correctly handle partial failures, timeout messages, and provider-specific error formats.
This is the single change most likely to improve hidden-scenario accuracy on non-standard error text.

**Priority 2: Full crash-resume workflow with SQLite**
The SQLite backend is implemented and checkpoints survive reconnect. The next step is implementing
full resume: calling `graph.invoke(None, config=config)` after process restart from the last
checkpoint. This requires saving the `thread_id` externally and integrating a resume CLI command.
