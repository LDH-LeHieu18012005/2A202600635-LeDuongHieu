"""Checkpointer adapter."""

from __future__ import annotations

from typing import Any


def build_checkpointer(kind: str = "memory", database_url: str | None = None) -> Any | None:  # noqa: ANN401
    """Return a LangGraph checkpointer.

    SQLite and Memory savers are supported.
    """
    if kind == "none":
        return None
    if kind == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()
    if kind == "sqlite":
        import sqlite3
        from pathlib import Path

        from langgraph.checkpoint.sqlite import SqliteSaver
        
        db_path = database_url or "outputs/checkpoints.db"
        if db_path.startswith("sqlite:///"):
            db_path = db_path[10:]
            
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        return SqliteSaver(conn)
    if kind == "postgres":
        raise NotImplementedError(
            "Postgres checkpointer is not configured (SQLite or Memory checkpointer recommended)."
        )
    raise ValueError(f"Unknown checkpointer kind: {kind}")
