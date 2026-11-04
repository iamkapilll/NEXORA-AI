"""SQLite persistence for checkpoints and conversation metadata."""
import sqlite3
import threading

from langgraph.checkpoint.sqlite import SqliteSaver

from backend.config import DOCUMENT_DB_PATH

document_conn = sqlite3.connect(
    database=str(DOCUMENT_DB_PATH),
    check_same_thread=False,
)
document_conn.execute(
    """
    CREATE TABLE IF NOT EXISTS conversation_documents (
        thread_id TEXT NOT NULL,
        document_id TEXT NOT NULL,
        filename TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        uploaded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (thread_id, document_id),
        UNIQUE (thread_id, content_hash)
    )
    """
)
document_conn.execute(
    "CREATE INDEX IF NOT EXISTS idx_conversation_documents_thread "
    "ON conversation_documents(thread_id)"
)
document_conn.execute(
    """
    CREATE TABLE IF NOT EXISTS conversation_titles (
        thread_id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        last_active_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """
)
document_conn.commit()
document_lock = threading.RLock()

checkpoint_conn = sqlite3.connect(
    database=str(DOCUMENT_DB_PATH),
    check_same_thread=False,
)
checkpoint = SqliteSaver(checkpoint_conn)
