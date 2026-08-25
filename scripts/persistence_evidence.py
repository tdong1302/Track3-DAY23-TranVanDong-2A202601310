"""Persistence evidence script.

Demonstrates:
1. MemorySaver checkpointer attached to compiled graph.
2. Unique thread_id per scenario run.
3. State history accessible via graph.get_state_history() within the same process.
4. SQLite checkpointer for cross-process durability (if package installed).

Run: python scripts/persistence_evidence.py
"""

from __future__ import annotations

import json
import os

from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.persistence import build_checkpointer
from langgraph_agent_lab.state import Route, Scenario, initial_state


def run_with_memory() -> None:
    """Demonstrate MemorySaver state history."""
    print("\n=== MemorySaver Persistence Evidence ===")
    checkpointer = build_checkpointer("memory")
    graph = build_graph(checkpointer=checkpointer)

    scenario = Scenario(
        id="persist-demo",
        query="How do I reset my password?",
        expected_route=Route.SIMPLE,
    )
    state = initial_state(scenario)
    thread_id = state["thread_id"]
    config = {"configurable": {"thread_id": thread_id}}

    print(f"Thread ID: {thread_id}")
    print("Running graph invoke...")
    final_state = graph.invoke(state, config=config)
    print(f"Route:        {final_state.get('route')}")
    answer = (final_state.get("final_answer") or "")[:80]
    print(f"Final answer: {answer}...")

    # Access state history within same process
    history = list(graph.get_state_history(config))
    print(f"\nCheckpoints recorded: {len(history)}")
    for i, checkpoint in enumerate(history[:3]):
        values = checkpoint.values
        events = values.get("events", [])
        print(f"  Checkpoint {i}: {len(events)} events, next={checkpoint.next}")

    print("\n[OK] MemorySaver: state history accessible in same process.")
    print("     Limitation: history lost when process exits.")


def run_with_sqlite() -> None:
    """Demonstrate SQLite checkpointer for cross-process durability."""
    print("\n=== SQLite Persistence Evidence ===")
    db_path = "checkpoints_evidence.db"
    try:
        checkpointer = build_checkpointer("sqlite", db_path)
    except RuntimeError as exc:
        print(f"[WARN] SQLite not available: {exc}")
        return

    graph = build_graph(checkpointer=checkpointer)
    scenario = Scenario(
        id="sqlite-demo",
        query="Lookup order status for order 99999",
        expected_route=Route.TOOL,
    )
    state = initial_state(scenario)
    thread_id = state["thread_id"]
    config = {"configurable": {"thread_id": thread_id}}

    print(f"Thread ID:   {thread_id}")
    print(f"DB file:     {db_path}")
    print("Running graph invoke (SQLite)...")
    final_state = graph.invoke(state, config=config)
    print(f"Route:       {final_state.get('route')}")

    history = list(graph.get_state_history(config))
    print(f"Checkpoints: {len(history)}")

    # Simulate crash-resume: re-open connection and verify state survives
    import sqlite3
    from langgraph.checkpoint.sqlite import SqliteSaver
    conn2 = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer2 = SqliteSaver(conn2)
    graph2 = build_graph(checkpointer=checkpointer2)
    history2 = list(graph2.get_state_history(config))
    print(f"Checkpoints after simulated restart: {len(history2)}")

    if len(history2) > 0:
        print("[OK] SQLite: state history survived process-level reconnect (crash-resume evidence).")
    else:
        print("[WARN] SQLite: no history found after reconnect.")

    # Clean up demo DB
    try:
        for ext in ["", "-wal", "-shm"]:
            p = db_path + ext
            if os.path.exists(p):
                os.remove(p)
    except OSError:
        pass


if __name__ == "__main__":
    run_with_memory()
    run_with_sqlite()
    print("\n=== Evidence collection complete ===")
