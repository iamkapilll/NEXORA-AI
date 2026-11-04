"""Persistent conversation titles and activity ordering."""
from datetime import datetime, timezone
import re
from langchain_core.messages import BaseMessage, HumanMessage
from backend.memory import checkpoint, document_conn, document_lock


def get_thread_documents(thread_id: str) -> list[dict[str, str]]:
    with document_lock:
        rows = document_conn.execute(
            """
            SELECT document_id, filename, uploaded_at
            FROM conversation_documents
            WHERE thread_id = ?
            ORDER BY uploaded_at, document_id
            """,
            (thread_id,)
        ).fetchall()
    return [
        {
            "document_id": row[0],
            "filename": row[1],
            "uploaded_at": row[2],
        }
        for row in rows
    ]


GREETING_ONLY_PATTERN = re.compile(
    r"^(?:hi|hello|hey|good morning|good afternoon|good evening)"
    r"(?:[,.! ]+(?:(?:i am|i'm|my name is)\b[\w'-]+)?)?[,.! ]*$",
    re.IGNORECASE
)

INTRODUCTION_ONLY_PATTERN = re.compile(
    r"^(?:i am|i'm|my name is)\s+[\w'-]+[,.! ]*$",
    re.IGNORECASE
)

TITLE_PREFIX_PATTERN = re.compile(
    r"^(?:can you tell me|could you tell me|i want to know|"
    r"please|tell me|show me|explain|describe|what (?:is|are)|"
    r"who is|give me|how (?:do|does|did|is|are))\s+",
    re.IGNORECASE
)

POSSESSIVE_TOPIC_PATTERN = re.compile(
    r"\b([a-z][a-z-]*)s(?=\s+(?:experience|experiences|skill|skills|"
    r"education|project|projects|resume|résumé|background|career)\b)",
    re.IGNORECASE
)

TITLE_ACRONYMS = {
    "ai": "AI",
    "api": "API",
    "cv": "CV",
    "pdf": "PDF",
    "rag": "RAG",
    "sql": "SQL",
}

def build_conversation_title(messages: list[BaseMessage]) -> str:
    for message in messages:
        if not isinstance(message, HumanMessage) or not isinstance(message.content, str):
            continue

        text = " ".join(message.content.split()).strip()
        if (
            not text
            or GREETING_ONLY_PATTERN.fullmatch(text)
            or INTRODUCTION_ONLY_PATTERN.fullmatch(text)
        ):
            continue

        title = TITLE_PREFIX_PATTERN.sub("", text).strip(" .?!,:;")
        if not title:
            title = text.strip(" .?!,:;")

        title = POSSESSIVE_TOPIC_PATTERN.sub(r"\1's", title)
        words = title.split()
        title = " ".join(
            TITLE_ACRONYMS.get(word.casefold(), word.capitalize())
            for word in words
        )
        if len(title) > 55:
            truncated = title[:52].rsplit(" ", 1)[0]
            title = f"{truncated}..."
        return title or "New Chat"

    return "New Chat"

def _normalize_activity_time(value: str | None = None) -> str:
    if value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc).isoformat()
        except ValueError:
            pass
    return datetime.now(timezone.utc).isoformat()

def save_conversation_title(
    thread_id: str,
    messages: list[BaseMessage],
    last_active_at: str | None = None
) -> str:
    candidate_title = build_conversation_title(messages)
    activity_time = _normalize_activity_time(last_active_at)

    with document_lock:
        existing = document_conn.execute(
            "SELECT title, last_active_at FROM conversation_titles "
            "WHERE thread_id = ?",
            (thread_id,)
        ).fetchone()

        if existing:
            current_title, current_activity = existing
            current_title = current_title.replace("''s", "'s")
            title = (
                candidate_title
                if current_title == "New Chat" and candidate_title != "New Chat"
                else current_title
            )
            latest_activity = max(
                current_activity,
                activity_time
            )
            document_conn.execute(
                """
                UPDATE conversation_titles
                SET title = ?, last_active_at = ?, updated_at = ?
                WHERE thread_id = ?
                """,
                (title, latest_activity, datetime.now(timezone.utc).isoformat(), thread_id)
            )
        else:
            title = candidate_title
            document_conn.execute(
                """
                INSERT INTO conversation_titles
                    (thread_id, title, last_active_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    thread_id,
                    title,
                    activity_time,
                    datetime.now(timezone.utc).isoformat()
                )
            )
        document_conn.commit()

    return title

def get_conversation_title(thread_id: str) -> str:
    with document_lock:
        row = document_conn.execute(
            "SELECT title FROM conversation_titles WHERE thread_id = ?",
            (thread_id,)
        ).fetchone()
    return row[0] if row else "New Chat"

def sync_conversation_titles() -> None:
    from backend.graph import chatbot

    for thread_id in get_all_threads():
        state = chatbot.get_state(
            config={"configurable": {"thread_id": thread_id}}
        )
        messages = state.values.get("messages", [])
        activity_time = state.created_at
        if not activity_time:
            documents = get_thread_documents(thread_id)
            if documents:
                activity_time = max(
                    document["uploaded_at"] for document in documents
                )
            elif get_conversation_title(thread_id) != "New Chat":
                continue

        save_conversation_title(
            thread_id,
            messages,
            activity_time
        )

def get_all_threads():
    activity_by_thread = {}
    for ckpt in checkpoint.list(None):
        thread_id = ckpt.config['configurable']['thread_id']
        activity_by_thread.setdefault(thread_id, "")
        created_at = getattr(ckpt, "created_at", None)
        if created_at:
            activity_time = _normalize_activity_time(created_at)
            activity_by_thread[thread_id] = max(
                activity_by_thread.get(thread_id, ""),
                activity_time
            )

    with document_lock:
        saved_titles = document_conn.execute(
            "SELECT thread_id, last_active_at FROM conversation_titles"
        ).fetchall()
        document_threads = document_conn.execute(
            """
            SELECT thread_id, MAX(uploaded_at)
            FROM conversation_documents
            GROUP BY thread_id
            """
        ).fetchall()

    for thread_id, last_active_at in saved_titles:
        activity_by_thread[thread_id] = max(
            activity_by_thread.get(thread_id, ""),
            last_active_at
        )
    for thread_id, uploaded_at in document_threads:
        activity_by_thread[thread_id] = max(
            activity_by_thread.get(thread_id, ""),
            _normalize_activity_time(uploaded_at)
        )

    return sorted(
        activity_by_thread,
        key=activity_by_thread.get,
        reverse=True
    )
