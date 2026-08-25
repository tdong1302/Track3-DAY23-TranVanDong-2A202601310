"""Checkpointer adapter."""

from __future__ import annotations

from typing import Any


def build_checkpointer(kind: str = "memory", database_url: str | None = None) -> Any | None:
    """Return a LangGraph checkpointer.

    Supported backends:
    - memory   : MemorySaver (default, in-process only)
    - sqlite   : SqliteSaver with WAL mode (durable, survives process restart)
    - postgres : AsyncPostgresSaver (optional extension, requires Docker)
    - none     : No checkpointer (stateless, no persistence)
    """
    if kind == "none":
        return None

    if kind == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()

    if kind == "sqlite":
        import sqlite3

        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError as exc:
            raise RuntimeError(
                "Install langgraph-checkpoint-sqlite: "
                "pip install langgraph-checkpoint-sqlite"
            ) from exc

        db_path = database_url or "checkpoints.db"
        conn = sqlite3.connect(db_path, check_same_thread=False)
        # Enable WAL mode for better concurrent read/write performance
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.commit()
        return SqliteSaver(conn)

    if kind == "postgres":
        raise NotImplementedError(
            "Postgres checkpointer is an optional extension. "
            "Use Docker Compose to start the DB, then implement AsyncPostgresSaver."
        )

    raise ValueError(f"Unknown checkpointer kind: {kind!r}")
